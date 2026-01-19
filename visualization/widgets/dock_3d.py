import pyqtgraph.opengl as gl
import pyqtgraph as pg
from PySide6.QtGui import QColor, QVector3D
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QSlider, QCheckBox, QLabel
from PySide6.QtCore import Qt, Signal
import numpy as np

from visualization.widgets.interactive_view import InteractiveGLViewWidget


class View3DDockWidget(QWidget):
    sig_frame_request = Signal(int)
    sig_edit_wind = Signal()
    sig_edit_wave = Signal()
    sig_edit_mooring = Signal()
    sig_edit_structure = Signal()

    def __init__(self, params):
        super().__init__()
        self.params = params

        # 几何参数 (基于中心点定位)
        self.h_plat_bot = -20.0
        self.h_plat_top = 10.0
        self.h_tp_top   = 12.0
        self.h_hub      = self.params.HubHeight # 90.0m

        # 机舱尺寸
        self.nac_len = 15.0
        self.nac_w = 5.0
        self.nac_h = 5.0

        # 高度链计算
        self.z_nac_bottom = self.h_hub - (self.nac_h / 2.0) # 87.5m

        self.len_main_col = self.h_plat_top - self.h_plat_bot
        self.len_tp       = self.h_tp_top - self.h_plat_top
        self.len_tower    = self.z_nac_bottom - self.h_tp_top

        # 中心坐标
        self.z_cen_main  = (self.h_plat_top + self.h_plat_bot) / 2.0
        self.z_cen_tp    = (self.h_tp_top + self.h_plat_top) / 2.0
        self.z_cen_tower = self.h_tp_top + (self.len_tower / 2.0)
        self.z_cen_nac   = self.h_hub

        # 材质库
        self.col_sky = QColor(120, 170, 210)
        self.mat_white = (0.96, 0.96, 0.98, 1.0)
        self.mat_red   = (0.85, 0.15, 0.15, 1.0)
        self.mat_dark  = (0.15, 0.18, 0.2, 1.0)
        self.mat_plat  = (0.85, 0.75, 0.2, 1.0)
        self.mat_tp    = (0.2, 0.2, 0.25, 1.0)
        self.mat_fairlead = (1.0, 0.5, 0.0, 1.0)

        self.setup_ui()
        self.init_scene()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.view = InteractiveGLViewWidget()
        self.view.setBackgroundColor(self.col_sky)
        self.view.setCameraPosition(distance=180, elevation=15, azimuth=-140)
        layout.addWidget(self.view)

        self.view.clicked_sky.connect(self.sig_edit_wind.emit)
        self.view.clicked_wave.connect(self.sig_edit_wave.emit)
        self.view.clicked_mooring.connect(self.sig_edit_mooring.emit)
        self.view.clicked_structure.connect(self.sig_edit_structure.emit)

        ctrl_bar = QHBoxLayout(); ctrl_bar.setContentsMargins(5, 5, 5, 5)
        self.chk_sync = QCheckBox("Live Sync"); self.chk_sync.setChecked(True)
        self.lbl_time = QLabel("T: 0.00s"); self.lbl_time.setFixedWidth(70)
        self.slider = QSlider(Qt.Horizontal); self.slider.setRange(0, 0)
        self.slider.valueChanged.connect(self.on_slider_changed)
        ctrl_bar.addWidget(self.chk_sync); ctrl_bar.addWidget(self.lbl_time); ctrl_bar.addWidget(self.slider)
        layout.addLayout(ctrl_bar)

    def _create_cylinder(self, r_base, r_top, length, color, rows=12, cols=24):
        """创建中心位于原点的圆柱体"""
        md = gl.MeshData.cylinder(rows=rows, cols=cols, radius=[r_base, r_top], length=length)
        verts = md.vertexes()
        verts[:, 2] -= length / 2.0
        md.setVertexes(verts)
        return gl.GLMeshItem(meshdata=md, smooth=True, color=color, shader='shaded')

    def _create_box_mesh(self, size_x, size_y, size_z, color):
        verts = np.array([
            [0,0,0], [1,0,0], [1,1,0], [0,1,0],
            [0,0,1], [1,0,1], [1,1,1], [0,1,1]
        ], dtype=np.float32)
        verts[:,0] *= size_x; verts[:,1] *= size_y; verts[:,2] *= size_z
        verts -= np.array([size_x/2, size_y/2, size_z/2])
        faces = np.array([[0,1,2], [0,2,3], [4,5,6], [4,6,7], [0,4,7], [0,7,3], [1,5,6], [1,6,2], [0,1,5], [0,5,4], [3,2,6], [3,6,7]])
        md = gl.MeshData(vertexes=verts, faces=faces)
        return gl.GLMeshItem(meshdata=md, smooth=False, color=color, shader='shaded', computeNormals=True)

    def _create_blade_mesh(self, length):
        """
        [V2.11 New] 带红色叶尖的真实感叶片
        """
        num_sections = 25 # 增加分段以获得更好的颜色过渡
        r_nodes = np.linspace(0, length, num_sections)
        verts = []
        faces = []
        colors = [] # 顶点颜色数组

        for i, r in enumerate(r_nodes):
            rel_r = r / length

            # 几何参数
            if rel_r < 0.05: chord = 3.5
            elif rel_r < 0.3: chord = 4.5
            else: chord = 4.5 * (1 - (rel_r - 0.3)/0.7 * 0.8)

            twist_deg = 13.0 * (1 - rel_r)
            twist = np.radians(twist_deg)
            thickness = chord * (0.2 if rel_r > 0.2 else 1.0)
            if rel_r < 0.05: thickness = chord

            # 颜色决定: 尖端 15% 为红色
            section_color = self.mat_red if rel_r > 0.85 else self.mat_white

            # 构建截面
            c, s = np.cos(twist), np.sin(twist)
            p_le = np.array([0, chord*0.3])
            p_te = np.array([0, -chord*0.7])
            p_up = np.array([thickness*0.5, -chord*0.2])
            p_lo = np.array([-thickness*0.5, -chord*0.2])

            def transform(p_2d):
                y_new = p_2d[1] * c - p_2d[0] * s
                z_new = p_2d[1] * s + p_2d[0] * c
                return [r, y_new, z_new]

            base_idx = i * 4
            verts.append(transform(p_le)); colors.append(section_color)
            verts.append(transform(p_up)); colors.append(section_color)
            verts.append(transform(p_te)); colors.append(section_color)
            verts.append(transform(p_lo)); colors.append(section_color)

            if i > 0:
                p, cur = idx-4, idx
                for j in range(4):
                    faces.append([p+j, cur+j, cur+(j+1)%4])
                    faces.append([p+j, cur+(j+1)%4, p+(j+1)%4])
            idx = base_idx + 4 # Update for next loop

        md = gl.MeshData(vertexes=np.array(verts), faces=np.array(faces), vertexColors=np.array(colors))
        return gl.GLMeshItem(meshdata=md, smooth=True, shader='shaded', computeNormals=True)

    def init_scene(self):
        axis = gl.GLAxisItem(); axis.setSize(30, 30, 30); self.view.addItem(axis)

        # --- 1. Platform ---
        self.plat_parts = []

        # Main Column
        self.main_col = self._create_cylinder(r_base=3.25, r_top=3.25, length=self.len_main_col, color=self.mat_plat)
        self.plat_parts.append({'item': self.main_col, 'local_z': self.z_cen_main, 'local_xy': (0,0)})
        self.view.addItem(self.main_col)

        # Offset Columns
        r_off = 28.87
        len_off = 32.0; z_cen_off = -4.0

        for ang in [0, 120, 240]:
            rad = np.radians(ang)
            x, y = r_off*np.cos(rad), r_off*np.sin(rad)
            col = self._create_cylinder(r_base=6.0, r_top=6.0, length=len_off, color=self.mat_plat)
            self.plat_parts.append({'item': col, 'local_z': z_cen_off, 'local_xy': (x,y)})
            self.view.addItem(col)

            brace_pts = np.array([[0,0,-5], [x,y,-5]])
            brace = gl.GLLinePlotItem(pos=brace_pts, width=4, color=self.mat_dark)
            self.plat_parts.append({'item': brace, 'type': 'brace', 'p_local': brace_pts})
            self.view.addItem(brace)

        # --- 2. TP & Tower ---
        self.tp = self._create_cylinder(r_base=3.25, r_top=3.00, length=self.len_tp, color=self.mat_tp)
        self.view.addItem(self.tp)

        self.tower = self._create_cylinder(r_base=3.00, r_top=1.93, length=self.len_tower, color=self.mat_white)
        self.view.addItem(self.tower)

        # --- 3. RNA ---
        self.nacelle = self._create_box_mesh(self.nac_len, self.nac_w, self.nac_h, self.mat_white)
        self.view.addItem(self.nacelle)

        self.hub = self._create_cylinder(r_base=2.0, r_top=2.0, length=4.0, color=self.mat_dark)
        self.hub.rotate(90, 0, 1, 0)
        self.view.addItem(self.hub)

        # [V2.11] 使用新的红白叶片
        self.blades = []
        for i in range(3):
            b = self._create_blade_mesh(self.params.R)
            self.view.addItem(b); self.blades.append(b)

        # --- 4. Mooring System ---
        self.lines = []
        for i in range(3):
            # 加粗线条，提高可见度
            l = gl.GLLinePlotItem(width=3.0, color=self.mat_dark, antialias=True)
            self.view.addItem(l); self.lines.append(l)

        d = self.params.WaterDepth
        self.anchors = [QVector3D(837*np.cos(r), 837*np.sin(r), -d) for r in [np.pi, np.pi/3, -np.pi/3]]
        self.fairleads_local = [QVector3D(34.0*np.cos(r), 34.0*np.sin(r), -14) for r in [np.pi, np.pi/3, -np.pi/3]]

        # [V2.11 New] 导缆孔实体模型 (Fairlead Geometry)
        # 在立柱上增加显眼的连接点
        self.fairlead_items = []
        for fl_local in self.fairleads_local:
            # 创建一个小方块或球体作为连接点
            item = self._create_box_mesh(2.0, 2.0, 2.0, self.mat_fairlead)
            self.view.addItem(item)
            # 存储其局部坐标以便跟随运动
            self.fairlead_items.append({'item': item, 'local_pos': fl_local})

        # --- 5. Wave ---
        self.wave_x = np.linspace(-300, 300, 80); self.wave_y = np.linspace(-300, 300, 80)
        self.wave_X, self.wave_Y = np.meshgrid(self.wave_x, self.wave_y)
        self.wave = gl.GLSurfacePlotItem(x=self.wave_x, y=self.wave_y, z=np.zeros_like(self.wave_X), shader='shaded', computeNormals=True, smooth=True)
        self.view.addItem(self.wave)

    def on_slider_changed(self, val):
        if not self.chk_sync.isChecked(): self.sig_frame_request.emit(val)

    def update_timeline(self, total, curr):
        self.slider.setMaximum(total - 1)
        if self.chk_sync.isChecked():
            self.slider.blockSignals(True); self.slider.setValue(total - 1); self.slider.blockSignals(False)
        self.lbl_time.setText(f"T: {curr:.2f}s")

    def _compute_wave_colors(self, Z):
        z_min, z_max = Z.min(), Z.max()
        z_norm = (Z - z_min) / (z_max - z_min + 1e-5)
        deep = np.array([0.0, 0.25, 0.45], dtype=np.float32)
        foam = np.array([0.7, 0.85, 0.95], dtype=np.float32)
        col = (1-z_norm[...,None])*deep + z_norm[...,None]*foam
        alpha = np.ones_like(Z)[...,None]
        return np.concatenate([col, alpha], axis=-1)

    def update_pose(self, state, t_time):
        surge, sway, heave = state[0], state[1], state[2]
        roll, pitch, yaw = state[3], state[4], state[5]

        # 1. Base Motion
        tr_base = pg.Transform3D()
        tr_base.translate(surge, sway, heave)
        tr_base.rotate(yaw, 0, 0, 1); tr_base.rotate(pitch, 0, 1, 0); tr_base.rotate(roll, 1, 0, 0)

        # 2. Hull
        for part in self.plat_parts:
            if 'local_z' in part:
                tr = pg.Transform3D(tr_base)
                x, y = part['local_xy']
                tr.translate(x, y, part['local_z'])
                part['item'].setTransform(tr)
            elif 'type' in part and part['type'] == 'brace':
                pts = part['p_local']
                p1 = tr_base.map(QVector3D(pts[0,0], pts[0,1], pts[0,2]))
                p2 = tr_base.map(QVector3D(pts[1,0], pts[1,1], pts[1,2]))
                part['item'].setData(pos=np.array([[p1.x(),p1.y(),p1.z()], [p2.x(),p2.y(),p2.z()]]))

        # 3. TP & Tower
        tr_tp = pg.Transform3D(tr_base); tr_tp.translate(0, 0, self.z_cen_tp)
        self.tp.setTransform(tr_tp)
        tr_tower = pg.Transform3D(tr_base); tr_tower.translate(0, 0, self.z_cen_tower)
        self.tower.setTransform(tr_tower)

        # 4. RNA
        tr_nac = pg.Transform3D(tr_base); tr_nac.translate(0, 0, self.z_cen_nac); tr_nac.translate(-4, 0, 0)
        self.nacelle.setTransform(tr_nac)
        tr_hub = pg.Transform3D(tr_nac); tr_hub.translate(self.nac_len/2.0 + 1.0, 0, 0); tr_hub.rotate(90, 0, 1, 0)
        self.hub.setTransform(tr_hub)

        azimuth = (t_time * 12.1 * 6.0) % 360.0
        for i, b in enumerate(self.blades):
            tr_b = pg.Transform3D(tr_hub)
            tr_b.rotate(azimuth + i*120, 0, 0, 1)
            tr_b.translate(0, 0, 2.5); tr_b.rotate(0, 0, 1, 0)
            b.setTransform(tr_b)

        # 5. Mooring & Fairleads
        cr, sr = np.cos(np.radians(roll)), np.sin(np.radians(roll))
        cp, sp = np.cos(np.radians(pitch)), np.sin(np.radians(pitch))
        cy, sy = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
        R = np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr], [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr], [-sp, cp*sr, cp*cr]])

        # Update Fairlead Geometry [V2.11 New]
        for fl_data in self.fairlead_items:
            tr_fl = pg.Transform3D(tr_base)
            loc = fl_data['local_pos']
            tr_fl.translate(loc.x(), loc.y(), loc.z())
            fl_data['item'].setTransform(tr_fl)

        # Update Lines
        for i, line in enumerate(self.lines):
            fl = np.array([self.fairleads_local[i].x(), self.fairleads_local[i].y(), self.fairleads_local[i].z()])
            fl_glo = np.dot(R, fl) + np.array([surge, sway, heave])
            line.setData(pos=np.array([[fl_glo[0], fl_glo[1], fl_glo[2]], [self.anchors[i].x(), self.anchors[i].y(), self.anchors[i].z()]]))

        # 6. Wave
        self.wave.resetTransform()
        phase = t_time * 1.5
        Z = self.params.Env_WaveHeight * np.sin(0.05 * self.wave_X - phase)
        self.wave.setData(z=Z, colors=self._compute_wave_colors(Z))
