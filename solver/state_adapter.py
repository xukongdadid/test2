# HHU-OWT-Sim/solver/state_adapter.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
import math


def rad2deg(x: float) -> float:
    return x * 180.0 / math.pi


@dataclass
class AdapterConfig:
    # 结构求解出来的角度是否是 rad（通常是）
    angles_are_rad: bool = True


class StateAdapter:
    """
    把 v2 worker 输出的 data(dict) 转成可视化期望的通道键：
    - dof_*  -> q*
    - env_*  -> env_wave_elev / env_wind_speed / thrust / gen_torque
    同时保留原始字段（不破坏 3D 视图使用的 dof_*）
    """

    def __init__(self, cfg: AdapterConfig | None = None) -> None:
        self.cfg = cfg or AdapterConfig()

    def adapt(self, data: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = dict(data)  # 先保留原始所有键

        # ---------- 1) DOF 映射：dof_* -> q* ----------
        # 平台 6DOF
        self._map(out, data, "dof_Surge", "qSg", angle=False)
        self._map(out, data, "dof_Sway",  "qSw", angle=False)
        self._map(out, data, "dof_Heave", "qHv", angle=False)
        self._map(out, data, "dof_Roll",  "qR",  angle=True)
        self._map(out, data, "dof_Pitch", "qP",  angle=True)
        self._map(out, data, "dof_Yaw",   "qY",  angle=True)

        # 塔架 4 个模态
        self._map(out, data, "dof_TwrFA1", "qTFA1", angle=False)
        self._map(out, data, "dof_TwrFA2", "qTFA2", angle=False)
        self._map(out, data, "dof_TwrSS1", "qTSS1", angle=False)
        self._map(out, data, "dof_TwrSS2", "qTSS2", angle=False)

        # 机舱&传动 3
        self._map(out, data, "dof_NacYaw", "qyaw",  angle=True)
        self._map(out, data, "dof_GenAz",  "qGeAz", angle=True)
        self._map(out, data, "dof_DrTr",   "qDrTr", angle=True)

        # 叶片 3*3（你 CHANNEL_CONFIG 里只配了 F1/E1/F2）
        self._map(out, data, "dof_B1F1", "qB1F1", angle=False)
        self._map(out, data, "dof_B1E1", "qB1E1", angle=False)
        self._map(out, data, "dof_B1F2", "qB1F2", angle=False)

        self._map(out, data, "dof_B2F1", "qB2F1", angle=False)
        self._map(out, data, "dof_B2E1", "qB2E1", angle=False)
        self._map(out, data, "dof_B2F2", "qB2F2", angle=False)

        self._map(out, data, "dof_B3F1", "qB3F1", angle=False)
        self._map(out, data, "dof_B3E1", "qB3E1", angle=False)
        self._map(out, data, "dof_B3F2", "qB3F2", angle=False)

        # ---------- 2) 环境/输出映射：env_* -> 可视化通道键 ----------
        # 你控制面板期望 env_wind_speed / env_wave_elev
        wind_key = "env_WindSpeed" if "env_WindSpeed" in data else "WindSpeed"
        wave_key = "env_WaveElev" if "env_WaveElev" in data else "WaveElev"
        if wind_key in data:
            out["env_wind_speed"] = float(data[wind_key])
        if wave_key in data:
            out["env_wave_elev"] = float(data[wave_key])
        if "CurrentSpeed" in data:
            out["env_current_speed"] = float(data["CurrentSpeed"])

        # 你控制面板还配置了 thrust / gen_torque
        # v2 worker 里叫 env_AeroThrust / env_GenTorque
        thrust_key = "env_AeroThrust" if "env_AeroThrust" in data else "AeroThrust"
        torque_key = "env_GenTorque" if "env_GenTorque" in data else "GenTq"
        if thrust_key in data:
            out["thrust"] = float(data[thrust_key])
        if torque_key in data:
            out["gen_torque"] = float(data[torque_key])

        # （可选）保证通道一定存在，防止 UI 勾选后 key 不存在报错/空图
        self._fill_defaults(out)

        return out

    def _map(self, out: Dict[str, Any], data: Dict[str, Any],
             src_key: str, dst_key: str, angle: bool) -> None:
        if src_key not in data:
            return
        v = float(data[src_key])
        if angle and self.cfg.angles_are_rad:
            v = rad2deg(v)
        out[dst_key] = v

    def _fill_defaults(self, out: Dict[str, Any]) -> None:
        defaults = [
            "qSg","qSw","qHv","qR","qP","qY",
            "qTFA1","qTFA2","qTSS1","qTSS2",
            "qyaw","qGeAz","qDrTr",
            "qB1F1","qB1E1","qB1F2",
            "qB2F1","qB2E1","qB2F2",
            "qB3F1","qB3E1","qB3F2",
            "env_wind_speed","env_wave_elev",
            "env_current_speed",
            "thrust","gen_torque",
        ]
        for k in defaults:
            out.setdefault(k, 0.0)
