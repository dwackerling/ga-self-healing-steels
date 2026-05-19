from __future__ import annotations

import contextlib
import math
import os
from dataclasses import dataclass
from typing import Any

from tc_python import TCPython, CompositionUnit, ThermodynamicQuantity


@dataclass
class ThermoCalcBackend:
    thermo_database: str
    mobility_database: str
    elements: list[str]
    matrix_phase: str = "BCC_A2"
    precip_phase: str = "C14_LAVES"
    suppress_tc_output: bool = True

    def __post_init__(self):
        self._tc_context = None
        self.session = None
        self.system = None
        self.calc_eq = None
        self.calc_df = None

    @staticmethod
    @contextlib.contextmanager
    def _suppress_output(enabled: bool = True):
        """
        No-op output context.

        Do not redirect stdout/stderr here: global stream redirection is not
        thread-safe when scikit-learn/joblib workers are active.
        TC-Python verbosity should be controlled through logging/warnings.
        """
        yield

    def __enter__(self):
        self._tc_context = TCPython()

        with self._suppress_output(self.suppress_tc_output):
            self.session = self._tc_context.__enter__()

            builder = self.session.select_thermodynamic_and_kinetic_databases_with_elements(
                self.thermo_database,
                self.mobility_database,
                self.elements,
            )

            self.system = builder.get_system()

            self.calc_eq = self.system.with_single_equilibrium_calculation()

            self.calc_df = (
                self.system
                .with_property_model_calculation("Driving force")
                .set_composition_unit(CompositionUnit.MASS_PERCENT)
            )

        return self

    def __exit__(self, exc_type, exc, tb):
        if self._tc_context is not None:
            with self._suppress_output(self.suppress_tc_output):
                self._tc_context.__exit__(exc_type, exc, tb)

        self._tc_context = None
        self.session = None
        self.system = None
        self.calc_eq = None
        self.calc_df = None

    @staticmethod
    def _normalise_element_symbol(element: str) -> str:
        e = str(element).strip()
        if not e:
            raise ValueError("Empty element symbol.")
        return e[0].upper() + e[1:].lower()

    @staticmethod
    def _safe_float(value: Any, default=0.0):
        try:
            x = float(value)
        except Exception:
            return default

        if not math.isfinite(x):
            return default

        return x

    def _set_equilibrium_conditions(
        self,
        composition_wt: dict[str, float],
        temperature_c: float,
    ) -> None:
        """
        Set all non-Fe components explicitly before each equilibrium calculation.

        Thermo-Calc calculation objects are stateful. Therefore, all elements in
        self.elements except Fe are reset every time, including zero values.
        """
        if self.calc_eq is None:
            raise RuntimeError("ThermoCalcBackend is not open.")

        comp_norm = {
            self._normalise_element_symbol(k): float(v)
            for k, v in composition_wt.items()
            if v is not None
        }

        total_non_fe = 0.0

        for element in self.elements:
            el = self._normalise_element_symbol(element)
            if el.upper() == "FE":
                continue

            wt = float(comp_norm.get(el, 0.0))
            if wt < 0.0:
                raise ValueError(f"Negative composition for {el}: {wt}")

            total_non_fe += wt
            self.calc_eq.set_condition(
                ThermodynamicQuantity.mass_fraction_of_a_component(el),
                wt / 100.0,
            )

        if total_non_fe >= 100.0:
            raise ValueError(
                f"Invalid composition: non-Fe sum is {total_non_fe:.6f} wt%."
            )

        self.calc_eq.set_condition(
            ThermodynamicQuantity.temperature(),
            float(temperature_c) + 273.15,
        )

    def _safe_phase_fraction(self, result, phase_name: str) -> float:
        try:
            value = result.get_value_of(
                ThermodynamicQuantity.volume_fraction_of_a_phase(phase_name)
            )
            return self._safe_float(value, default=0.0)
        except Exception:
            return 0.0

    def calculate_equilibrium_result(
        self,
        composition_wt: dict[str, float],
        temperature_c: float,
    ):
        """
        Return raw Thermo-Calc equilibrium result after setting conditions.

        Used to extract phase fractions, phase compositions and diffusion
        quantities from the same equilibrium state.
        """
        self._set_equilibrium_conditions(
            composition_wt=composition_wt,
            temperature_c=temperature_c,
        )

        with self._suppress_output(self.suppress_tc_output):
            return self.calc_eq.calculate()

    def calculate_equilibrium_snapshot(
        self,
        composition_wt: dict[str, float],
        temperature_c: float,
        phases: dict[str, str],
    ) -> dict[str, float]:
        """
        Return selected phase fractions at a given temperature.
        """
        result = self.calculate_equilibrium_result(
            composition_wt=composition_wt,
            temperature_c=temperature_c,
        )

        matrix_bcc = phases["matrix_bcc"]
        matrix_fcc = phases["matrix_fcc"]
        liquid = phases["liquid"]
        laves = phases["laves"]
        m23c6 = phases["m23c6"]
        mx = phases["mx"]

        snapshot = {
            "LIQUID": self._safe_phase_fraction(result, liquid),
            "BCC_A2": self._safe_phase_fraction(result, matrix_bcc),
            "FCC_A1": self._safe_phase_fraction(result, matrix_fcc),
            "LAVES": self._safe_phase_fraction(result, laves),
            "M23C6": self._safe_phase_fraction(result, m23c6),
            "MX": self._safe_phase_fraction(result, mx),
        }

        snapshot["MATRIX_TOTAL_BCC_FCC"] = snapshot["BCC_A2"] + snapshot["FCC_A1"]
        return snapshot

    def get_phase_mole_fractions(
        self,
        result,
        phase_name: str,
        elements: list[str],
    ) -> dict[str, float | None]:
        """
        Extract mole fractions of selected elements in a phase.

        Returns None if Thermo-Calc cannot provide the value.
        """
        out: dict[str, float | None] = {}

        for element in elements:
            el = self._normalise_element_symbol(element)
            try:
                value = result.get_value_of(
                    ThermodynamicQuantity.composition_of_phase_as_mole_fraction(
                        phase_name,
                        el,
                    )
                )
                out[el] = self._safe_float(value, default=None)
            except Exception:
                out[el] = None

        return out

    def get_phase_weight_fractions(
        self,
        result,
        phase_name: str,
        elements: list[str],
    ) -> dict[str, float | None]:
        """
        Extract weight fractions of selected elements in a phase.

        Returns None if Thermo-Calc cannot provide the value.
        """
        out: dict[str, float | None] = {}

        for element in elements:
            el = self._normalise_element_symbol(element)
            try:
                value = result.get_value_of(
                    ThermodynamicQuantity.composition_of_phase_as_weight_fraction(
                        phase_name,
                        el,
                    )
                )
                out[el] = self._safe_float(value, default=None)
            except Exception:
                out[el] = None

        return out

    def get_diffusivities(
        self,
        result,
        phase_name: str,
        elements: list[str],
        dependent_element: str = "Fe",
    ) -> dict[str, float | None]:
        """
        Extract tracer diffusivities in the target phase.

        Falls back to diagonal chemical diffusivity if tracer diffusivity is
        unavailable.

        Values are returned in the units provided by Thermo-Calc.
        """
        out: dict[str, float | None] = {}

        dep = self._normalise_element_symbol(dependent_element)

        for element in elements:
            el = self._normalise_element_symbol(element)

            tracer = None
            try:
                value = result.get_value_of(
                    ThermodynamicQuantity.tracer_diffusion_coefficient(
                        phase_name,
                        el,
                    )
                )
                tracer = self._safe_float(value, default=None)
            except Exception:
                tracer = None

            if tracer is not None:
                out[el] = tracer
                continue

            chemical = None
            try:
                value = result.get_value_of(
                    ThermodynamicQuantity.chemical_diffusion_coefficient(
                        phase_name,
                        el,
                        el,
                        dep,
                    )
                )
                chemical = self._safe_float(value, default=None)
            except Exception:
                chemical = None

            out[el] = chemical

        return out

    def calculate_driving_force(
        self,
        composition_wt: dict[str, float],
        temperature_c: float,
        precip_phase: str | None = None,
        fe_symbol: str = "Fe",
    ) -> float:
        """
        Calculate the Thermo-Calc normalized driving force for precipitation.

        Uses the Thermo-Calc property model:
            "Driving force"

        Composition unit:
            MASS_PERCENT

        Fe is kept as the dependent element and must not be explicitly set.

        Returns
        -------
        float
            normalizedDrivingForce as returned by TC-Python.
        """
        if self.calc_df is None:
            raise RuntimeError("ThermoCalcBackend is not open.")

        precip = precip_phase or self.precip_phase
        T_K = float(temperature_c) + 273.15

        comp_norm = {
            self._normalise_element_symbol(k): float(v)
            for k, v in composition_wt.items()
            if v is not None
        }

        total_non_fe = 0.0

        for element in self.elements:
            el = self._normalise_element_symbol(element)
            if el.upper() == fe_symbol.upper():
                continue

            wt = float(comp_norm.get(el, 0.0))
            if wt < 0.0:
                raise ValueError(f"Negative composition for {el}: {wt}")

            total_non_fe += wt

        if total_non_fe > 100.0 + 1.0e-9:
            raise ValueError(
                f"Invalid composition: non-Fe sum is {total_non_fe:.6f} wt%."
            )

        self.calc_df.set_temperature(T_K)

        # Set all non-Fe components explicitly, including zeros.
        # Fe remains the dependent component.
        for element in self.elements:
            el = self._normalise_element_symbol(element)
            if el.upper() == fe_symbol.upper():
                continue

            wt = float(comp_norm.get(el, 0.0))
            self.calc_df.set_composition(el, wt)

        with self._suppress_output(self.suppress_tc_output):
            result = (
                self.calc_df
                .set_argument("precipitate", precip)
                .calculate()
            )

        return float(result.get_value_of("normalizedDrivingForce"))