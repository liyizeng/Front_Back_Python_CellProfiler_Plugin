import numpy as np
from cellprofiler_core.module import Module
from cellprofiler_core.object import Objects
from cellprofiler_core.setting.subscriber import LabelSubscriber
from cellprofiler_core.setting.text import Float, Integer, LabelName

__doc__ = """
LeadingTrailingHalves
=====================

Splits migrating objects into leading and trailing halves
based on the direction of motion determined from centroid
movement between consecutive frames.

Utilizes a Smoothed Moving Average (SMMA) to calculate continuous 
migration trajectories while filtering out sub-threshold 
noise based on a given micrometer resolution.
"""

class LeadingTrailingHalves(Module):

    module_name = 'LeadingTrailingHalves'
    category = 'Object Processing'
    variable_revision_number = 3

    def create_settings(self):
        self.object_name = LabelSubscriber(
            'Select tracked objects',
            'None',
        )
        
        self.leading_name = LabelName(
            'Name the leading half objects',
            'LeadingHalf'
        )

        self.trailing_name = LabelName(
            'Name the trailing half objects',
            'TrailingHalf'
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
            self.leading_name,
            self.trailing_name,
            self.pixel_scale,
            self.min_displacement,
            self.smoothing_window
        ]

    def run(self, workspace):
        input_objects = workspace.object_set.get_objects(self.object_name.value)
        labels = input_objects.segmented

        pixels_per_micron = 1.0 / self.pixel_scale.value
        d_min = self.min_displacement.value * pixels_per_micron
        
        n = self.smoothing_window.value

        leading_labels = np.zeros_like(labels)
        trailing_labels = np.zeros_like(labels)

        object_ids = np.unique(labels)
        object_ids = object_ids[object_ids != 0]

        if getattr(self, 'explored_cell', None) is None:
            self.explored_cell = {}

        current_explored_cell = {}

        for obj_id in object_ids:
            y, x = np.where(labels == obj_id)

            if x.size == 0:
                continue
            
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
                        migration_direction = (D_unitvec + (n - 1) * last_direction) / n # SMMA
                        migration_direction = migration_direction / np.linalg.norm(migration_direction)

            current_explored_cell[obj_id] = {
                'prev_centroid': current_centroid,
                'last_direction': migration_direction
            }

            if migration_direction is None:
                continue
            
            pixel_coord = np.column_stack((x, y))
            pixel_vector = pixel_coord - current_centroid
            proj = np.dot(pixel_vector, migration_direction) # projection of pixel vector onto migration direction unit vector
            
            front = proj >= 0
            back = proj < 0

            leading_labels[y[front], x[front]] = obj_id
            trailing_labels[y[back], x[back]] = obj_id

        self.explored_cell = current_explored_cell

        leading_objects = Objects()
        leading_objects.segmented = leading_labels
        trailing_objects = Objects()
        trailing_objects.segmented = trailing_labels
        workspace.object_set.add_objects(leading_objects, self.leading_name.value)
        workspace.object_set.add_objects(trailing_objects, self.trailing_name.value)

        if self.show_window:
            workspace.display_data.input_labels = labels
            workspace.display_data.leading_labels = leading_labels
            workspace.display_data.trailing_labels = trailing_labels

    def display(self, workspace, figure):
        figure.set_subplots((1,3))
        figure.subplot_imshow_labels(0, 0, workspace.display_data.input_labels, 'Input')
        figure.subplot_imshow_labels(0, 1, workspace.display_data.leading_labels, 'Leading Half')
        figure.subplot_imshow_labels(0, 2, workspace.display_data.trailing_labels, 'Trailing Half')