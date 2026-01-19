import os
import numpy as np
from PySide6.QtCore import Qt, Signal, QEvent
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QSlider, QCheckBox, QLabel
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.vtkCommonTransforms import vtkTransform
from vtkmodules.vtkFiltersSources import (
    vtkCylinderSource,
    vtkCubeSource,
    vtkPlaneSource,
    vtkLineSource,
    vtkSphereSource,
)
from vtkmodules.vtkIOGeometry import vtkOBJReader, vtkGLTFReader
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkRenderer,
    vtkSkybox,
    vtkTexture,
    vtkPropPicker,
)
from vtkmodules.vtkRenderingOpenGL2 import vtkOpenGLPolyDataMapper
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkIOImage import vtkHDRReader, vtkPNGReader, vtkJPEGReader


class VTKViewDockWidget(QWidget):
    sig_frame_request = Signal(int)
    sig_edit_wind = Signal()
    sig_edit_wave = Signal()
    sig_edit_mooring = Signal()
    sig_edit_structure = Signal()

    def __init__(self, params):
        super().__init__()
        self.params = params
        self._last_pose = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self._prop_roles = {}
        self._structure_actors = []
        self._mooring_lines = []
        self._mooring_sources = []
        self._wind_source = None
        self._water_actor = None
        self._water_mapper = None
        self._water_time = 0.0
        self._right_press_pos = None
        self._right_dragging = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        layout.addWidget(self.vtk_widget)
        self._build_timeline_controls(layout)

        self.renderer = vtkRenderer()
        self.render_window = self.vtk_widget.GetRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.interactor = self.render_window.GetInteractor()
        self.interactor.SetInteractorStyle(vtkInteractorStyleTrackballCamera())

        self.vtk_widget.installEventFilter(self)
        self._init_scene()
        self._init_visual_state()
        self.interactor.Initialize()

    def _build_timeline_controls(self, layout):
        ctrl_bar = QHBoxLayout()
        ctrl_bar.setContentsMargins(5, 5, 5, 5)
        self.chk_sync = QCheckBox("Live Sync")
        self.chk_sync.setChecked(True)
        self.lbl_time = QLabel("T: 0.00s")
        self.lbl_time.setFixedWidth(70)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.valueChanged.connect(self.on_slider_changed)
        ctrl_bar.addWidget(self.chk_sync)
        ctrl_bar.addWidget(self.lbl_time)
        ctrl_bar.addWidget(self.slider)
        layout.addLayout(ctrl_bar)

    def eventFilter(self, obj, event):
        if obj is self.vtk_widget and event.type() == QEvent.MouseButtonDblClick:
            self._handle_double_click(event)
            return True
        if obj is self.vtk_widget and event.type() == QEvent.MouseButtonPress:
            if event.button() == Qt.RightButton:
                self._right_press_pos = event.position() if hasattr(event, "position") else event.localPos()
                self._right_dragging = False
        if obj is self.vtk_widget and event.type() == QEvent.MouseMove:
            if self._right_press_pos is not None and event.buttons() & Qt.RightButton:
                pos = event.position() if hasattr(event, "position") else event.localPos()
                if (pos - self._right_press_pos).manhattanLength() > 4:
                    self._right_dragging = True
        if obj is self.vtk_widget and event.type() == QEvent.MouseButtonRelease:
            if event.button() == Qt.RightButton and self._right_press_pos is not None:
                if not self._right_dragging:
                    self._handle_double_click(event)
                    self._right_press_pos = None
                    self._right_dragging = False
                    return True
                self._right_press_pos = None
                self._right_dragging = False
        return super().eventFilter(obj, event)

    def _handle_double_click(self, event):
        pos = event.position() if hasattr(event, "position") else event.localPos()
        x = int(pos.x())
        y = int(self.vtk_widget.height() - pos.y())
        picker = vtkPropPicker()
        picker.Pick(x, y, 0, self.renderer)
        prop = picker.GetViewProp()
        role = self._prop_roles.get(prop)
        if role == "wave":
            self.sig_edit_wave.emit()
            return
        if role == "mooring":
            self.sig_edit_mooring.emit()
            return
        if role == "structure":
            self.sig_edit_structure.emit()
            return
        self.sig_edit_wind.emit()

    def _init_scene(self):
        self.renderer.SetBackground(0.06, 0.09, 0.12)
        self._init_skybox()
        self._init_water()
        self._init_turbine_model()
        self._init_mooring()
        self._init_wind_arrow()
        self.renderer.ResetCamera()

    def _init_skybox(self):
        skybox_path = getattr(self.params, "Visual_SkyboxPath", "")
        if not skybox_path or not os.path.exists(skybox_path):
            return
        skybox = vtkSkybox()
        texture = vtkTexture()
        ext = os.path.splitext(skybox_path)[1].lower()
        if ext == ".hdr":
            reader = vtkHDRReader()
            reader.SetFileName(skybox_path)
        elif ext == ".png":
            reader = vtkPNGReader()
            reader.SetFileName(skybox_path)
        else:
            reader = vtkJPEGReader()
            reader.SetFileName(skybox_path)
        reader.Update()
        texture.SetInputConnection(reader.GetOutputPort())
        texture.MipmapOn()
        texture.InterpolateOn()
        skybox.SetTexture(texture)
        self.renderer.AddActor(skybox)
        self._prop_roles[skybox] = "wind"

    def _apply_pbr(self, actor, base_color, metallic=0.05, roughness=0.4):
        prop = actor.GetProperty()
        if hasattr(prop, "SetInterpolationToPBR"):
            prop.SetInterpolationToPBR()
        if hasattr(prop, "SetBaseColor"):
            prop.SetBaseColor(*base_color)
        else:
            prop.SetColor(*base_color)
        if hasattr(prop, "SetMetallic"):
            prop.SetMetallic(metallic)
        if hasattr(prop, "SetRoughness"):
            prop.SetRoughness(roughness)

    def _init_water(self):
        plane = vtkPlaneSource()
        plane.SetXResolution(200)
        plane.SetYResolution(200)
        plane.SetOrigin(-400, -400, 0)
        plane.SetPoint1(400, -400, 0)
        plane.SetPoint2(-400, 400, 0)
        plane.Update()

        mapper = vtkOpenGLPolyDataMapper()
        mapper.SetInputConnection(plane.GetOutputPort())
        actor = vtkActor()
        actor.SetMapper(mapper)
        self._apply_pbr(actor, (0.04, 0.2, 0.35), metallic=0.02, roughness=0.15)

        shader_prop = actor.GetShaderProperty()
        shader_prop.AddVertexShaderReplacement(
            "//VTK::PositionVC::Impl",
            True,
            "//VTK::PositionVC::Impl\nvec4 pos = vertexMC;\n"
            "float waveTime = 0.0;\n"
            "float wave = sin(pos.x * 0.03 + waveTime) * 0.6 + cos(pos.y * 0.02 + waveTime) * 0.4;\n"
            "pos.z += wave;\n"
            "gl_Position = MCDCMatrix * pos;\n",
            False,
        )
        self._water_mapper = mapper
        self._water_actor = actor
        self.renderer.AddActor(actor)
        self._prop_roles[actor] = "wave"

    def _init_turbine_model(self):
        model_path = getattr(self.params, "Visual_TurbineModelPath", "")
        if model_path and os.path.exists(model_path):
            ext = os.path.splitext(model_path)[1].lower()
            if ext in {".gltf", ".glb"}:
                reader = vtkGLTFReader()
                reader.SetFileName(model_path)
                reader.Update()
                mapper = vtkPolyDataMapper()
                mapper.SetInputConnection(reader.GetOutputPort())
                actor = vtkActor()
                actor.SetMapper(mapper)
                self._apply_pbr(actor, (0.85, 0.87, 0.9), metallic=0.1, roughness=0.35)
                self.renderer.AddActor(actor)
                self._structure_actors.append(actor)
                self._prop_roles[actor] = "structure"
                return
            if ext == ".obj":
                reader = vtkOBJReader()
                reader.SetFileName(model_path)
                reader.Update()
                mapper = vtkPolyDataMapper()
                mapper.SetInputConnection(reader.GetOutputPort())
                actor = vtkActor()
                actor.SetMapper(mapper)
                self._apply_pbr(actor, (0.85, 0.87, 0.9), metallic=0.1, roughness=0.35)
                self.renderer.AddActor(actor)
                self._structure_actors.append(actor)
                self._prop_roles[actor] = "structure"
                return

        tower = vtkCylinderSource()
        tower.SetRadius(3.0)
        tower.SetHeight(80.0)
        tower.SetResolution(40)
        tower_mapper = vtkPolyDataMapper()
        tower_mapper.SetInputConnection(tower.GetOutputPort())
        tower_actor = vtkActor()
        tower_actor.SetMapper(tower_mapper)
        tower_actor.SetPosition(0, 0, 40)
        self._apply_pbr(tower_actor, (0.9, 0.9, 0.92), metallic=0.05, roughness=0.3)
        self.renderer.AddActor(tower_actor)

        nacelle = vtkCubeSource()
        nacelle.SetXLength(12.0)
        nacelle.SetYLength(5.0)
        nacelle.SetZLength(4.0)
        nacelle_mapper = vtkPolyDataMapper()
        nacelle_mapper.SetInputConnection(nacelle.GetOutputPort())
        nacelle_actor = vtkActor()
        nacelle_actor.SetMapper(nacelle_mapper)
        nacelle_actor.SetPosition(6, 0, 82)
        self._apply_pbr(nacelle_actor, (0.8, 0.82, 0.85), metallic=0.08, roughness=0.25)
        self.renderer.AddActor(nacelle_actor)

        hub = vtkSphereSource()
        hub.SetRadius(2.5)
        hub.SetThetaResolution(32)
        hub.SetPhiResolution(32)
        hub_mapper = vtkPolyDataMapper()
        hub_mapper.SetInputConnection(hub.GetOutputPort())
        hub_actor = vtkActor()
        hub_actor.SetMapper(hub_mapper)
        hub_actor.SetPosition(13, 0, 82)
        self._apply_pbr(hub_actor, (0.2, 0.2, 0.22), metallic=0.2, roughness=0.4)
        self.renderer.AddActor(hub_actor)

        for angle in (0, 120, 240):
            blade = vtkCylinderSource()
            blade.SetRadius(0.6)
            blade.SetHeight(50.0)
            blade.SetResolution(24)
            blade_mapper = vtkPolyDataMapper()
            blade_mapper.SetInputConnection(blade.GetOutputPort())
            blade_actor = vtkActor()
            blade_actor.SetMapper(blade_mapper)
            blade_actor.SetPosition(13, 0, 82)
            blade_actor.RotateX(90)
            blade_actor.RotateZ(angle)
            blade_actor.AddPosition(0, 0, 25)
            self._apply_pbr(blade_actor, (0.9, 0.9, 0.92), metallic=0.05, roughness=0.4)
            self.renderer.AddActor(blade_actor)
            self._structure_actors.append(blade_actor)

        self._structure_actors.extend([tower_actor, nacelle_actor, hub_actor])
        for actor in self._structure_actors:
            self._prop_roles[actor] = "structure"

    def _init_mooring(self):
        self._mooring_sources = []
        self._mooring_lines = []
        fairleads = getattr(self.params, "Moor_FairleadPoints", [])
        anchors = getattr(self.params, "Moor_AnchorPoints", [])
        for fairlead, anchor in zip(fairleads, anchors):
            line = vtkLineSource()
            line.SetPoint1(*fairlead)
            line.SetPoint2(*anchor)
            mapper = vtkPolyDataMapper()
            mapper.SetInputConnection(line.GetOutputPort())
            actor = vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(0.25, 0.25, 0.25)
            actor.GetProperty().SetLineWidth(2.5)
            self.renderer.AddActor(actor)
            self._mooring_sources.append(line)
            self._mooring_lines.append(actor)
            self._prop_roles[actor] = "mooring"

    def _init_wind_arrow(self):
        line = vtkLineSource()
        line.SetPoint1(0, -140, 60)
        line.SetPoint2(40, -140, 60)
        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(line.GetOutputPort())
        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1.0, 1.0, 1.0)
        actor.GetProperty().SetLineWidth(3.0)
        self.renderer.AddActor(actor)
        self._wind_source = line
        self._prop_roles[actor] = "wind"

    def _init_visual_state(self):
        self.update_pose([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.0)
        self.update_environment({"env_wind_speed": self.params.Env_WindSpeed})
        self.render_window.Render()

    def update_pose(self, state, t_time):
        surge, sway, heave = state[0], state[1], state[2]
        roll, pitch, yaw = state[3], state[4], state[5]
        self._last_pose = (surge, sway, heave, roll, pitch, yaw)

        transform = vtkTransform()
        transform.Translate(surge, sway, heave)
        transform.RotateZ(yaw)
        transform.RotateY(pitch)
        transform.RotateX(roll)
        for actor in self._structure_actors:
            actor.SetUserTransform(transform)

        cr, sr = np.cos(np.radians(roll)), np.sin(np.radians(roll))
        cp, sp = np.cos(np.radians(pitch)), np.sin(np.radians(pitch))
        cy, sy = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
        R = np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ])
        fairleads = getattr(self.params, "Moor_FairleadPoints", [])
        anchors = getattr(self.params, "Moor_AnchorPoints", [])
        for idx, (line, fairlead, anchor) in enumerate(zip(self._mooring_sources, fairleads, anchors)):
            fl = np.array(fairlead)
            fl_glo = np.dot(R, fl) + np.array([surge, sway, heave])
            line.SetPoint1(*fl_glo)
            line.SetPoint2(*anchor)

        self._water_time = t_time
        self.render_window.Render()

    def update_environment(self, data):
        wind = float(data.get("env_wind_speed", 0.0))
        scale = max(20.0, min(100.0, wind * 4.0))
        if self._wind_source:
            self._wind_source.SetPoint1(0, -140, 60)
            self._wind_source.SetPoint2(scale, -140, 60)
        self.render_window.Render()

    def on_slider_changed(self, val):
        if not self.chk_sync.isChecked():
            self.sig_frame_request.emit(val)

    def update_timeline(self, total, curr):
        self.slider.setMaximum(max(0, total - 1))
        if self.chk_sync.isChecked():
            self.slider.blockSignals(True)
            self.slider.setValue(max(0, total - 1))
            self.slider.blockSignals(False)
        self.lbl_time.setText(f"T: {curr:.2f}s")

    def apply_params(self, params):
        self.params = params
        self.renderer.RemoveAllViewProps()
        self._prop_roles.clear()
        self._structure_actors = []
        self._mooring_lines = []
        self._mooring_sources = []
        self._wind_source = None
        self._water_actor = None
        self._water_mapper = None
        self._init_scene()
        self._init_visual_state()
