from dataclasses import dataclass, field, asdict
from typing import Optional
import json


@dataclass
class BatteryInfo:
    implant_date: Optional[str] = None
    voltage: Optional[str] = None
    current_ua: Optional[str] = None
    remaining_capacity_pct: Optional[str] = None
    estimated_longevity: Optional[str] = None
    rrt: Optional[str] = None


@dataclass
class LeadTestResult:
    pacing_threshold: Optional[str] = None
    pacing_threshold_date: Optional[str] = None
    sensing_mv: Optional[str] = None
    sensing_date: Optional[str] = None
    impedance_ohm: Optional[str] = None
    impedance_date: Optional[str] = None


@dataclass
class ProgrammedParameters:
    mode: Optional[str] = None
    base_rate: Optional[str] = None
    av_paced_ms: Optional[str] = None
    av_sensed_ms: Optional[str] = None
    upper_rate_limit: Optional[str] = None
    max_sensor_rate: Optional[str] = None
    rate_hysteresis: Optional[str] = None
    vip_enabled: Optional[str] = None
    rate_adaptive_av: Optional[str] = None
    rate_adaptive_pvarp: Optional[str] = None
    ventricular_safety_pacing: Optional[str] = None


@dataclass
class StimulationSettings:
    configuration: Optional[str] = None
    amplitude_v: Optional[str] = None
    pulse_width_ms: Optional[str] = None
    senseability: Optional[str] = None
    sensitivity_mv: Optional[str] = None


@dataclass
class Diagnostics:
    since: Optional[str] = None
    ap_paced_pct: Optional[str] = None
    vp_paced_pct: Optional[str] = None
    mode_switch_pct: Optional[str] = None
    ams_episodes: Optional[str] = None


@dataclass
class PacemakerReport:
    source_file: Optional[str] = None
    report_type: Optional[str] = None
    patient: Optional[str] = None
    device: Optional[str] = None
    facility: Optional[str] = None
    interrogation_date: Optional[str] = None
    visit_type: Optional[str] = None

    battery_ra: Optional[BatteryInfo] = None
    battery_rv: Optional[BatteryInfo] = None

    test_ra: Optional[LeadTestResult] = None
    test_rv: Optional[LeadTestResult] = None

    parameters: Optional[ProgrammedParameters] = None

    stimulation_ra: Optional[StimulationSettings] = None
    stimulation_rv: Optional[StimulationSettings] = None

    diagnostics: Optional[Diagnostics] = None

    raw_ocr_pages: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)
