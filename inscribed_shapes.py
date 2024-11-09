import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle as MplCircle, Rectangle as MplRectangle, Polygon as MplPolygon, Patch
from matplotlib import transforms
from itertools import permutations
import unittest

# Base Shape class
class Shape:
    def __init__(self, color='black', alpha=0.5, linewidth=2, facecolor='none'):
        self.color = color
        self.alpha = alpha
        self.linewidth = linewidth
        self.facecolor = facecolor
        self.parent = None  # Add parent reference

    def set_parent(self, parent):
        self.parent = parent

    def plot(self, ax):
        raise NotImplementedError("Subclasses should implement this method.")

    def get_vertices(self):
        raise NotImplementedError("Subclasses should implement this method.")

# Base Positioning System
class ShapePositioner:
    def get_position(self, shape, parent=None, grandparent=None):
        raise NotImplementedError()

    def get_vertices(self, shape, parent=None, grandparent=None):
        raise NotImplementedError()

class CirclePositioner(ShapePositioner):
    def get_position(self, shape, parent=None, grandparent=None):
        if isinstance(parent, Triangle):
            h = (np.sqrt(3) / 2) * parent.side_length
            
            if parent.rotate_base_down:
                # For case 6: Triangle with base down
                return (0, -h/3 + shape.radius)
            else:
                # For case 4: Triangle with base up
                if isinstance(parent.parent, Square):
                    S = parent.parent.side_length / 2
                    triangle_bottom_y = S - h
                    return (0, triangle_bottom_y + shape.radius)
                return (0, 0)
        elif isinstance(parent, Square):
            if isinstance(parent.parent, Triangle):
                h = (np.sqrt(3) / 2) * parent.parent.side_length
                return (0, -h/3 + parent.side_length/2)
            return (0, 0)
        return (0, 0)

    def get_vertices(self, shape, parent=None, grandparent=None):
        pos = self.get_position(shape, parent, grandparent)
        angles = np.linspace(0, 360, 9)[:-1]  # 0 to 315 degrees
        return [(shape.radius * np.cos(np.deg2rad(a)) + pos[0],
                shape.radius * np.sin(np.deg2rad(a)) + pos[1])
                for a in angles]

class SquarePositioner(ShapePositioner):
    def get_position(self, shape, parent=None, grandparent=None):
        if isinstance(parent, Triangle):
            h = (np.sqrt(3) / 2) * parent.side_length
            if parent.rotate_base_down:
                # For case 6: Triangle with base down
                return (0, -h/3)
            else:
                # For case 2: Triangle inscribed in circle
                if isinstance(parent.parent, Circle):
                    R = parent.parent.radius
                    return (0, -R * np.cos(np.pi/3))
                else:
                    return (0, -h/2)
        return (0, 0)

    def get_vertices(self, shape, parent=None, grandparent=None):
        pos = self.get_position(shape, parent, grandparent)
        if isinstance(parent, Triangle):
            h = (np.sqrt(3) / 2) * parent.side_length
            if parent.rotate_base_down:
                triangle_bottom_y = -h/3
            else:
                if isinstance(parent.parent, Circle):
                    R = parent.parent.radius
                    triangle_bottom_y = -R * np.cos(np.pi/3)
                else:
                    triangle_bottom_y = -h/2
            
            half_side = shape.side_length / 2
            square_bottom_y = triangle_bottom_y
            square_top_y = square_bottom_y + shape.side_length
            
            return [
                (-half_side, square_top_y),    # Top-left
                (half_side, square_top_y),     # Top-right
                (half_side, square_bottom_y),  # Bottom-right
                (-half_side, square_bottom_y)  # Bottom-left
            ]
        else:
            S = shape.side_length / 2
            return [
                (-S + pos[0], S + pos[1]),    # Top-left
                (S + pos[0], S + pos[1]),     # Top-right
                (S + pos[0], -S + pos[1]),    # Bottom-right
                (-S + pos[0], -S + pos[1])    # Bottom-left
            ]

class TrianglePositioner(ShapePositioner):
    def get_position(self, shape, parent=None, grandparent=None):
        return (0, 0)  # Position will be handled in get_vertices

    def get_vertices(self, shape, parent=None, grandparent=None):
        h = (np.sqrt(3) / 2) * shape.side_length

        if shape.rotate_base_down:
            # Base at bottom, for cases 5 and 6
            return [
                (0, 2*h/3),                      # Top vertex
                (-shape.side_length/2, -h/3),    # Bottom-left vertex
                (shape.side_length/2, -h/3)      # Bottom-right vertex
            ]
        elif isinstance(parent, Circle):
            # For case 2: vertices touch circle at 90°, 210°, and 330°
            R = parent.radius
            return [
                (0, R),                                    # Top vertex
                (-R * np.sin(np.pi/3), -R * np.cos(np.pi/3)),  # Bottom-left
                (R * np.sin(np.pi/3), -R * np.cos(np.pi/3))    # Bottom-right
            ]
        else:
            # Original orientation (base at top) for other cases
            S = shape.side_length / 2
            return [
                (-shape.side_length/2, S-h),    # Bottom-left vertex
                (shape.side_length/2, S-h),     # Bottom-right vertex
                (0, S)                          # Top vertex
            ]

# Circle Shape
class Circle(Shape):
    def __init__(self, radius, **kwargs):
        super().__init__(**kwargs)
        self.radius = radius
        self._positioner = CirclePositioner()

    def plot(self, ax):
        pos = self._positioner.get_position(self, self.parent, 
                                          self.parent.parent if self.parent else None)
        circle = MplCircle(pos, self.radius,
                         edgecolor=self.color,
                         facecolor=self.facecolor,
                         linewidth=self.linewidth,
                         alpha=self.alpha)
        ax.add_patch(circle)
        return circle

    def get_vertices(self):
        return self._positioner.get_vertices(self, self.parent,
                                           self.parent.parent if self.parent else None)

# Square Shape
class Square(Shape):
    def __init__(self, side_length, **kwargs):
        super().__init__(**kwargs)
        self.side_length = side_length
        self._positioner = SquarePositioner()

    def plot(self, ax):
        vertices = self._positioner.get_vertices(self, self.parent,
                                               self.parent.parent if self.parent else None)
        square = MplPolygon(vertices, closed=True,
                           edgecolor=self.color,
                           facecolor=self.facecolor,
                           linewidth=self.linewidth,
                           alpha=self.alpha)
        ax.add_patch(square)
        return square

    def get_vertices(self):
        return self._positioner.get_vertices(self, self.parent,
                                           self.parent.parent if self.parent else None)

# Triangle Shape
class Triangle(Shape):
    def __init__(self, side_length, rotate_base_down=False, **kwargs):
        super().__init__(**kwargs)
        self.side_length = side_length
        self.rotate_base_down = rotate_base_down
        self._positioner = TrianglePositioner()

    def plot(self, ax):
        vertices = self._positioner.get_vertices(self, self.parent,
                                               self.parent.parent if self.parent else None)
        triangle = MplPolygon(vertices, closed=True,
                            edgecolor=self.color,
                            facecolor=self.facecolor,
                            linewidth=self.linewidth,
                            alpha=self.alpha)
        ax.add_patch(triangle)
        return triangle

    def get_vertices(self):
        return self._positioner.get_vertices(self, self.parent,
                                           self.parent.parent if self.parent else None)

# ShapeHierarchy class to manage the hierarchy and plotting
class ShapeHierarchy:
    def __init__(self, hierarchy, subplot_pos, fig, colors=None, alphas=None):
        """
        Initialize the hierarchy with a list of shapes from outermost to innermost.

        Parameters:
        - hierarchy: List of shapes in order
        - subplot_pos: Tuple of (row, col, index) for subplot positioning
        - fig: Matplotlib figure object
        - colors: Dict mapping shapes to colors
        - alphas: Dict mapping shapes to alpha transparency
        """
        self.hierarchy = hierarchy
        self.colors = colors or {
            'circle': 'green',
            'square': 'red',
            'triangle': 'blue'
        }
        self.alphas = alphas or {
            'circle': 0.5,
            'square': 0.5,
            'triangle': 0.5
        }
        self.shape_sizes = {}
        self.fig = fig
        self.ax = fig.add_subplot(*subplot_pos)
        self.ax.set_aspect('equal')
        self.initial_size = 1.0  # This will be set from main()

    # Inscribing functions
    def inscribe_square_in_circle(self, radius):
        return radius * np.sqrt(2)

    def inscribe_circle_in_square(self, side_length):
        return side_length / 2

    def inscribe_triangle_in_square(self, side_length):
        """
        Calculate the side length of an equilateral triangle inscribed in a square,
        such that it touches the square at the middle of its left and right sides.
        The triangle's side length should equal the square's side length.
        """
        return side_length

    def inscribe_square_in_triangle(self, side_length):
        """
        Calculates the side length of a square inscribed in an equilateral triangle.
        Formula: S = (sqrt(3) * L) / (2 + sqrt(3))
        """
        return (np.sqrt(3) * side_length) / (2 + np.sqrt(3))

    def inscribe_triangle_in_circle(self, radius):
        """
        Calculates the side length of an equilateral triangle inscribed in a circle.
        Formula: T = 2 * R * sin(60°) = R * sqrt(3)
        """
        return 2 * radius * np.sin(np.pi / 3)  # 2R sin(60°) = R * sqrt(3)

    def inscribe_circle_in_triangle(self, side_length):
        """
        Calculates the radius of a circle inscribed in an equilateral triangle.
        Formula: r = (a * sqrt(3)) / 6
        """
        return (side_length * np.sqrt(3)) / 6  # r = (a√3)/6

    def calculate_inscribed_size(self, outer_shape, inner_shape, outer_size):
        """
        Calculate the size of the inner shape based on the outer shape.

        Parameters:
        - outer_shape: 'circle', 'square', or 'triangle'
        - inner_shape: 'circle', 'square', or 'triangle'
        - outer_size: size of the outer shape (radius for circle, side_length for square/triangle)

        Returns:
        - inner_size: size of the inner shape (radius for circle, side_length for square/triangle)
        """
        if outer_shape == inner_shape:
            raise ValueError("Outer shape and inner shape must be different.")

        if outer_shape == 'circle' and inner_shape == 'square':
            return self.inscribe_square_in_circle(outer_size)
        elif outer_shape == 'circle' and inner_shape == 'triangle':
            return self.inscribe_triangle_in_circle(outer_size)
        elif outer_shape == 'square' and inner_shape == 'circle':
            return self.inscribe_circle_in_square(outer_size)
        elif outer_shape == 'square' and inner_shape == 'triangle':
            return self.inscribe_triangle_in_square(outer_size)
        elif outer_shape == 'triangle' and inner_shape == 'circle':
            return self.inscribe_circle_in_triangle(outer_size)
        elif outer_shape == 'triangle' and inner_shape == 'square':
            return self.inscribe_square_in_triangle(outer_size)
        else:
            raise ValueError(f"Invalid inscribing relationship: {outer_shape} -> {inner_shape}")

    def plot(self):
        """
        Plot the hierarchy of shapes.
        """
        if not self.hierarchy:
            raise ValueError("Hierarchy list is empty.")

        # Calculate vertical offset if triangle is the outer shape
        y_offset = 0
        if self.hierarchy[0] == 'triangle':
            h = (np.sqrt(3) / 2) * self.initial_size
            y_offset = -h/3

        # Create transform for the offset
        offset_transform = self.ax.transData + transforms.ScaledTranslation(0, y_offset, self.fig.dpi_scale_trans)
        
        # Plot outermost shape
        outer_shape = self.hierarchy[0]
        shape_obj = self.create_shape(outer_shape, self.initial_size)
        patch = shape_obj.plot(self.ax)
        patch.set_transform(offset_transform)
        self.shape_sizes[outer_shape] = self.initial_size
        previous_shape = shape_obj

        # Plot subsequent shapes
        for i in range(1, len(self.hierarchy)):
            outer = self.hierarchy[i - 1]
            inner = self.hierarchy[i]
            outer_size = self.shape_sizes[outer]
            inner_size = self.calculate_inscribed_size(outer, inner, outer_size)
            self.shape_sizes[inner] = inner_size
            shape_obj = self.create_shape(inner, inner_size)
            shape_obj.set_parent(previous_shape)  # Set parent reference
            patch = shape_obj.plot(self.ax)
            patch.set_transform(offset_transform)
            previous_shape = shape_obj

        # Use consistent limits for all plots
        limit = 1.1
        self.ax.set_xlim(-limit, limit)
        self.ax.set_ylim(-limit, limit)
        self.ax.axis('off')

    def create_shape(self, shape_type, size):
        """
        Create a shape object based on the type and size.

        Parameters:
        - shape_type: 'circle', 'square', or 'triangle'
        - size: size of the shape (radius for circle, side_length for square/triangle)

        Returns:
        - Instance of the corresponding Shape subclass.
        """

        if shape_type == 'circle':
            return Circle(radius=size, color=self.colors['circle'], alpha=self.alphas['circle'])
        elif shape_type == 'square':
            return Square(side_length=size, color=self.colors['square'], alpha=self.alphas['square'])
        elif shape_type == 'triangle':
            # Rotate the triangle when it's the first shape in the hierarchy
            rotate_base_down = (shape_type == self.hierarchy[0])
            return Triangle(side_length=size, 
                      rotate_base_down=rotate_base_down,
                      color=self.colors['triangle'], 
                      alpha=self.alphas['triangle'])
        else:
            raise ValueError(f"Unsupported shape type: {shape_type}")

# Helper functions for point-in-shape tests
def point_in_circle(x, y, R):
    return (x**2 + y**2) <= R**2 + 1e-8  # Adding epsilon for floating point precision

def point_in_square(x, y, S):
    return (abs(x) <= S / 2 + 1e-8) and (abs(y) <= S / 2 + 1e-8)

def point_in_triangle(x, y, T):
    # Using barycentric coordinates for point-in-triangle test
    h = (np.sqrt(3) / 2) * T
    A = np.array([-T / 2, -h / 3])
    B = np.array([T / 2, -h / 3])
    C = np.array([0, 2 * h / 3])

    v0 = C - A
    v1 = B - A
    v2 = np.array([x, y]) - A

    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)

    denom = dot00 * dot11 - dot01 * dot01
    if denom == 0:
        return False  # Degenerate triangle

    u = (dot11 * dot02 - dot01 * dot12) / denom
    v = (dot00 * dot12 - dot01 * dot02) / denom

    return (u >= -1e-8) and (v >= -1e-8) and (u + v <= 1 + 1e-8)

# Unit Tests
class TestShapeHierarchy(unittest.TestCase):
    def setUp(self):
        self.hierarchy_manager = ShapeHierarchy(['circle', 'square', 'triangle'])

    def test_inscribe_square_in_circle(self):
        radius = 1.0
        expected_side = np.sqrt(2)
        calculated_side = self.hierarchy_manager.inscribe_square_in_circle(radius)
        self.assertAlmostEqual(calculated_side, expected_side, places=5, msg="Square inscribed in circle incorrect.")

    def test_inscribe_circle_in_square(self):
        side = np.sqrt(2)
        expected_radius = 0.70710678118  # sqrt(2)/2
        calculated_radius = self.hierarchy_manager.inscribe_circle_in_square(side)
        self.assertAlmostEqual(calculated_radius, expected_radius, places=5, msg="Circle inscribed in square incorrect.")

    def test_inscribe_triangle_in_square(self):
        side = 1.0  # Square's side_length
        expected_triangle_side = (np.sqrt(3) / 2) * side  # ≈0.8660
        calculated_triangle_side = self.hierarchy_manager.inscribe_triangle_in_square(side)
        self.assertAlmostEqual(calculated_triangle_side, expected_triangle_side, places=5, msg="Triangle inscribed in square incorrect.")

    def test_inscribe_circle_in_triangle(self):
        side = np.sqrt(2)
        expected_radius = (side * np.sqrt(3)) / 6  # (sqrt(2)*sqrt(3))/6 = sqrt(6)/6 ≈0.4082
        calculated_radius = self.hierarchy_manager.inscribe_circle_in_triangle(side)
        self.assertAlmostEqual(calculated_radius, expected_radius, places=5, msg="Circle inscribed in triangle incorrect.")

    def test_inscribe_triangle_in_circle(self):
        radius = 1.0
        expected_side = 2 * radius * np.sin(np.pi / 3)  # 2*sin(60)*1 = sqrt(3)≈1.7320
        calculated_side = self.hierarchy_manager.inscribe_triangle_in_circle(radius)
        self.assertAlmostEqual(calculated_side, np.sqrt(3), places=5, msg="Triangle inscribed in circle incorrect.")

    def test_inscribe_square_in_triangle(self):
        side = 1.0
        expected_side = (np.sqrt(3) * side) / (2 + np.sqrt(3))  # ≈0.464
        calculated_side = self.hierarchy_manager.inscribe_square_in_triangle(side)
        self.assertAlmostEqual(calculated_side, expected_side, places=5, msg="Square inscribed in triangle incorrect.")

    def test_invalid_inscribing_relationship(self):
        with self.assertRaises(ValueError):
            self.hierarchy_manager.calculate_inscribed_size('circle', 'circle', 1.0)

    def test_invalid_shape_type(self):
        with self.assertRaises(ValueError):
            self.hierarchy_manager.create_shape('hexagon', 1.0)

    def test_alignment_C_S_T(self):
        # Hierarchy: Circle > Square > Triangle
        hierarchy = ['circle', 'square', 'triangle']
        shape_hierarchy = ShapeHierarchy(hierarchy)
        # Calculate sizes
        R = 1.0
        S = shape_hierarchy.inscribe_square_in_circle(R)  # S = sqrt(2)
        T = shape_hierarchy.inscribe_triangle_in_square(S)  # T = (sqrt(3)/2)*S ≈0.8660 *1.41421.2247

        # Get Triangle vertices
        triangle = Triangle(T)
        vertices = triangle.get_vertices()

        # Check each vertex is within Square
        for x, y in vertices:
            self.assertTrue(point_in_square(x, y, S), f"Triangle vertex ({x}, {y}) not inside Square.")

        # Check Square vertices are within Circle
        square = Square(S)
        square_vertices = square.get_vertices()
        for x, y in square_vertices:
            self.assertTrue(point_in_circle(x, y, R), f"Square vertex ({x}, {y}) not inside Circle.")

    def test_alignment_C_T_S(self):
        # Hierarchy: Circle > Triangle > Square
        hierarchy = ['circle', 'triangle', 'square']
        shape_hierarchy = ShapeHierarchy(hierarchy)
        # Calculate sizes
        R = 1.0
        T = shape_hierarchy.inscribe_triangle_in_circle(R)  # T = sqrt(3)
        S = shape_hierarchy.inscribe_square_in_triangle(T)  # S ≈0.464

        # Get Square vertices
        square = Square(S)
        square_vertices = square.get_vertices()

        # Check each vertex is within Triangle
        for x, y in square_vertices:
            self.assertTrue(point_in_triangle(x, y, T), f"Square vertex ({x}, {y}) not inside Triangle.")

        # Check Triangle vertices are within Circle
        triangle = Triangle(T)
        triangle_vertices = triangle.get_vertices()
        for x, y in triangle_vertices:
            self.assertTrue(point_in_circle(x, y, R), f"Triangle vertex ({x}, {y}) not inside Circle.")

    def test_alignment_S_T_C(self):
        # Hierarchy: Square > Triangle > Circle
        hierarchy = ['square', 'triangle', 'circle']
        shape_hierarchy = ShapeHierarchy(hierarchy)
        # Calculate sizes
        S = 1.0
        T = shape_hierarchy.inscribe_triangle_in_square(S)  # T = (sqrt(3)/2)*S ≈0.8660
        R = shape_hierarchy.inscribe_circle_in_triangle(T)  # R = (T*sqrt(3))/6 ≈0.8660*1.7320/6≈0.25

        # Get Circle vertices
        circle = Circle(R)
        circle_vertices = circle.get_vertices()

        # Check each vertex is within Triangle
        for x, y in circle_vertices:
            self.assertTrue(point_in_triangle(x, y, T), f"Circle vertex ({x}, {y}) not inside Triangle.")

        # Check Triangle vertices are within Square
        triangle = Triangle(T)
        triangle_vertices = triangle.get_vertices()
        for x, y in triangle_vertices:
            self.assertTrue(point_in_square(x, y, S), f"Triangle vertex ({x}, {y}) not inside Square.")

    def test_alignment_T_S_C(self):
        # Hierarchy: Triangle > Square > Circle
        hierarchy = ['triangle', 'square', 'circle']
        shape_hierarchy = ShapeHierarchy(hierarchy)
        # Calculate sizes
        T = 1.0
        S = shape_hierarchy.inscribe_square_in_triangle(T)  # S ≈0.464
        R = shape_hierarchy.inscribe_circle_in_square(S)  # R=0.232

        # Get Circle vertices
        circle = Circle(R)
        circle_vertices = circle.get_vertices()

        # Check each vertex is within Square
        for x, y in circle_vertices:
            self.assertTrue(point_in_square(x, y, S), f"Circle vertex ({x}, {y}) not inside Square.")

        # Check Square vertices are within Triangle
        square = Square(S)
        square_vertices = square.get_vertices()
        for x, y in square_vertices:
            self.assertTrue(point_in_triangle(x, y, T), f"Square vertex ({x}, {y}) not inside Triangle.")

    def test_alignment_C_S_T_full(self):
        # Comprehensive test for Circle > Square > Triangle
        hierarchy = ['circle', 'square', 'triangle']
        shape_hierarchy = ShapeHierarchy(hierarchy)
        R = 1.0
        S = shape_hierarchy.inscribe_square_in_circle(R)  # S = sqrt(2)
        T = shape_hierarchy.inscribe_triangle_in_square(S)  # T = (sqrt(3)/2)*S ≈1.2247

        # Check Triangle vertices within Square
        triangle = Triangle(T)
        triangle_vertices = triangle.get_vertices()
        for x, y in triangle_vertices:
            self.assertTrue(point_in_square(x, y, S), f"Triangle vertex ({x}, {y}) not inside Square.")

        # Check Square vertices within Circle
        square = Square(S)
        square_vertices = square.get_vertices()
        for x, y in square_vertices:
            self.assertTrue(point_in_circle(x, y, R), f"Square vertex ({x}, {y}) not inside Circle.")

    def test_alignment_C_T_S_full(self):
        # Comprehensive test for Circle > Triangle > Square
        hierarchy = ['circle', 'triangle', 'square']
        shape_hierarchy = ShapeHierarchy(hierarchy)
        R = 1.0
        T = shape_hierarchy.inscribe_triangle_in_circle(R)  # T = sqrt(3)
        S = shape_hierarchy.inscribe_square_in_triangle(T)  # S ≈0.464

        # Check Square vertices within Triangle
        square = Square(S)
        square_vertices = square.get_vertices()
        for x, y in square_vertices:
            self.assertTrue(point_in_triangle(x, y, T), f"Square vertex ({x}, {y}) not inside Triangle.")

        # Check Triangle vertices within Circle
        triangle = Triangle(T)
        triangle_vertices = triangle.get_vertices()
        for x, y in triangle_vertices:
            self.assertTrue(point_in_circle(x, y, R), f"Triangle vertex ({x}, {y}) not inside Circle.")

    def test_alignment_S_T_C_full(self):
        # Comprehensive test for Square > Triangle > Circle
        hierarchy = ['square', 'triangle', 'circle']
        shape_hierarchy = ShapeHierarchy(hierarchy)
        S = 1.0
        T = shape_hierarchy.inscribe_triangle_in_square(S)  # T = (sqrt(3)/2)*S ≈0.8660
        R = shape_hierarchy.inscribe_circle_in_triangle(T)  # R ≈0.25

        # Check Circle vertices within Triangle
        circle = Circle(R)
        circle_vertices = circle.get_vertices()
        for x, y in circle_vertices:
            self.assertTrue(point_in_triangle(x, y, T), f"Circle vertex ({x}, {y}) not inside Triangle.")

        # Check Triangle vertices within Square
        triangle = Triangle(T)
        triangle_vertices = triangle.get_vertices()
        for x, y in triangle_vertices:
            self.assertTrue(point_in_square(x, y, S), f"Triangle vertex ({x}, {y}) not inside Square.")

    def test_alignment_T_S_C_full(self):
        # Comprehensive test for Triangle > Square > Circle
        hierarchy = ['triangle', 'square', 'circle']
        shape_hierarchy = ShapeHierarchy(hierarchy)
        T = 1.0
        S = shape_hierarchy.inscribe_square_in_triangle(T)  # S ≈0.464
        R = shape_hierarchy.inscribe_circle_in_square(S)  # R ≈0.232

        # Check Circle vertices within Square
        circle = Circle(R)
        circle_vertices = circle.get_vertices()
        for x, y in circle_vertices:
            self.assertTrue(point_in_square(x, y, S), f"Circle vertex ({x}, {y}) not inside Square.")

        # Check Square vertices within Triangle
        square = Square(S)
        square_vertices = square.get_vertices()
        for x, y in square_vertices:
            self.assertTrue(point_in_triangle(x, y, T), f"Square vertex ({x}, {y}) not inside Triangle.")

# Main execution
def main():
    """
    Main function to execute the script.
    """
    all_shapes = ['circle', 'square', 'triangle']
    possible_hierarchies = list(permutations(all_shapes))
    
    n_plots = len(possible_hierarchies)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(15, 5 * n_rows))
    
    custom_colors = {
        'circle': 'green',
        'square': 'red',
        'triangle': 'blue'
    }
    custom_alphas = {
        'circle': 0.5,
        'square': 0.5,
        'triangle': 0.5
    }

    # Scale factor to make shapes fit better in the canvas
    scale = 0.8  # Adjust this value to make shapes larger or smaller
    
    # Base sizes scaled down to fit better
    initial_sizes = {
        'circle': 1.0 * scale,  # radius
        'square': 2.0 * scale,  # side length
        'triangle': (2.0 * 2/np.sqrt(3)) * scale  # side length
    }

    for idx, hierarchy in enumerate(possible_hierarchies, 1):
        subplot_pos = (n_rows, n_cols, idx)
        shape_hierarchy = ShapeHierarchy(
            hierarchy, 
            subplot_pos, 
            fig,
            colors=custom_colors, 
            alphas=custom_alphas
        )
        
        outer_shape = hierarchy[0]
        shape_hierarchy.initial_size = initial_sizes[outer_shape]
        
        shape_hierarchy.plot()
        hierarchy_title = " > ".join([s.capitalize() for s in hierarchy])
        shape_hierarchy.ax.set_title(f"{idx}. {hierarchy_title}", fontsize=10)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Run unit tests
    print("Running unit tests...")
    unittest.main(argv=[''], exit=False)

    # Execute main plotting
    main()