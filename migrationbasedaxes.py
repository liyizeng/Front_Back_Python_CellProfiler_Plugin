import numpy as np
from cellprofiler_core.module import Module
from cellprofiler_core.setting.subscriber import LabelSubscriber
from cellprofiler_core.setting.text import Float, Integer

__doc__ = """
!!DRAFT!!

migration based more like migration cringe xdddd
"""

def label_text(label, x, y, txt):
    if x is not None and y is not None:
        label.text(
            x,
            y,
            txt,
            color='white',
            family='sans-serif',
            ha='center',
            va='center',
            fontsize=9
        )

def label_axes_text(ax, x, y, major_um, minor_um):
    label_text(ax, x, y, f'{major_um:.1f} / {minor_um:.1f} um')

def draw_axis_line(ax, centroid, direction, proj_min, proj_max, color):
    cx, cy = centroid
    x0, y0 = cx + direction[0] * proj_min, cy + direction[1] * proj_min
    x1, y1 = cx + direction[0] * proj_max, cy + direction[1] * proj_max
    ax.plot([x0, x1], [y0, y1], color=color, linewidth=1.5)

def axes_overlay(figure, workspace):
    ax = figure.subplot(0, 0)
    object_info = workspace.display_data.object_info

    for obj_id, info in object_info.items():
        cx, cy = info['cent']
        u = info['mig_dir']
        u_perp = np.array([-u[1], u[0]])
        label_axes_text(ax, cx, cy, info['major'], info['minor'])
        draw_axis_line(ax, info['cent'], u, info['major_min_px'], info['major_max_px'], 'red')
        draw_axis_line(ax, info['cent'], u_perp, info['minor_min_px'], info['minor_max_px'], 'cyan')

class MigrationBasedAxes(Module):

    module_name = 'MigrationBasedAxes'
    category = 'Measurement'
    variable_revision_number = 2

    def create_settings(self):
        self.object_name = LabelSubscriber(
            'Select objects to measure',
            'None'
        )

        self.pixel_scale = Float(
            'Image scale (micrometers per pixel)',
            0.75521,
            minval=0.0001,
            doc='For every pixel the object moves, how much does it displace IRL (in micrometers)?'
        )

        self.min_displacement = Float(
            'Minimum displacement threshold (micrometers)',
            1.00,
            minval=0.00,
            doc='Displacements smaller than this value are treated as noise.'
        )

        self.smoothing_window = Integer(
            'Smoothing window size',
            5,
            minval=1,
            doc='Decides how influential the past is over the new/current direction vector.'
        )

        self.explored_cell = None

    def settings(self):
        return [
            self.object_name,
            self.pixel_scale,
            self.min_displacement,
            self.smoothing_window
        ]

    def run(self, workspace):
        objects = workspace.object_set.get_objects(self.object_name.value)
        labels = objects.segmented

        pixels_per_micron = 1.0 / self.pixel_scale.value
        d_min = self.min_displacement.value * pixels_per_micron
        n = self.smoothing_window.value

        object_ids = np.unique(labels)
        object_ids = np.sort(object_ids[object_ids != 0])

        if getattr(self, 'explored_cell', None) is None:
            self.explored_cell = {}

        current_explored_cell = {}
        object_info = {}

        major = np.full(len(object_ids), np.nan)
        minor = np.full(len(object_ids), np.nan)

        for i, obj_id in enumerate(object_ids):
            y, x = np.where(labels == obj_id)
            
            if x.size == 0: continue

            current_centroid = np.array([x.mean(), y.mean()])
            migration_direction = None

            if obj_id in self.explored_cell:
                prev_centroid = self.explored_cell[obj_id]['prev_centroid']
                D_vec = current_centroid - prev_centroid
                D_mag = np.linalg.norm(D_vec)

                if D_mag < d_min:
                    migration_direction = self.explored_cell[obj_id]['last_direction']
                else:
                    D_unitvec = D_vec / D_mag
                    last_direction = self.explored_cell[obj_id]['last_direction']

                    if last_direction is None:
                        migration_direction = D_unitvec
                    else:
                        migration_direction = (D_unitvec + (n - 1) * last_direction) / n  # SMMA
                        migration_direction = migration_direction / np.linalg.norm(migration_direction)

            current_explored_cell[obj_id] = {
                'prev_centroid': current_centroid,
                'last_direction': migration_direction
            }

            if migration_direction is None: continue

            u = migration_direction
            u_perp = np.array([-u[1], u[0]])

            pixel_vector = np.column_stack((x, y)) - current_centroid
            proj_along = np.dot(pixel_vector, u)
            proj_perp = np.dot(pixel_vector, u_perp)
            
            along_min, along_max = proj_along.min(), proj_along.max()
            perp_min, perp_max = proj_perp.min(), proj_perp.max()

            length_along_px = proj_along.max() - proj_along.min()
            width_perp_px = proj_perp.max() - proj_perp.min()

            length_along_um = length_along_px * self.pixel_scale.value
            width_perp_um = width_perp_px * self.pixel_scale.value

            major[i] = length_along_um
            minor[i] = width_perp_um
            
            object_info[obj_id] = {
                'cent': current_centroid,
                'mig_dir': u,
                'major_min_px': along_min,
                'major_max_px': along_max,
                'minor_min_px': perp_min,
                'minor_max_px': perp_max,
                'major': length_along_um,
                'minor': width_perp_um,
                }
            
        self.explored_cell = current_explored_cell

        m = workspace.measurements
        m.add_measurement(self.object_name.value, 'MajorAxisLength', major)
        m.add_measurement(self.object_name.value, 'MinorAxisLength', minor)

        if self.show_window:
            workspace.display_data.labels = labels
            workspace.display_data.major = major
            workspace.display_data.minor = minor
            workspace.display_data.object_info = object_info

    def display(self, workspace, figure):
        figure.set_subplots((1, 1))
        figure.subplot_imshow_labels(0, 0, workspace.display_data.labels, 'Tracked Objects')
        axes_overlay(figure, workspace)
        
    def get_categories(self, pipeline, object_name):
        if object_name == self.object_name.value:
            return ['MigrationBasedAxes']
        return []

    def get_measurements(self, pipeline, object_name, category):
        if object_name == self.object_name.value and category == 'MigrationBasedAxes':
            return ['MajorAxisLength', 'MinorAxisLength']
        return []