from typing import List, Dict, Tuple, Optional, Any, Union
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle as MplCircle, Rectangle as MplRectangle, Polygon as MplPolygon, Patch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib import transforms
from itertools import permutations
import unittest

# Base Shape class
class Shape:
    def __init__(self, color: str = 'black', alpha: float = 0.5, 
                 linewidth: float = 2, facecolor: str = 'none') -> None:
        self.color: str = color
        self.alpha: float = alpha
        self.linewidth: float = linewidth
        self.facecolor: str = facecolor
        self.parent: Optional['Shape'] = None

    def set_parent(self, parent: 'Shape') -> None:
        self.parent = parent

    def plot(self, ax: Axes) -> Patch:
        raise NotImplementedError("Subclasses should implement this method.")

    def get_vertices(self) -> List[Tuple[float, float]]:
        raise NotImplementedError("Subclasses should implement this method.")

# Base Positioning System
class ShapePositioner:
    def get_position(self, shape: Shape, 
                    parent: Optional[Shape] = None, 
                    grandparent: Optional[Shape] = None) -> Tuple[float, float]:
        raise NotImplementedError()

    def get_vertices(self, shape: Shape, 
                    parent: Optional[Shape] = None,
                    grandparent: Optional[Shape] = None) -> List[Tuple[float, float]]:
        raise NotImplementedError()

class CirclePositioner(ShapePositioner):
    def get_position(self, shape: 'Circle', 
                    parent: Optional[Shape] = None,
                    grandparent: Optional[Shape] = None) -> Tuple[float, float]:
        if isinstance(parent, Triangle):
            h: float = (np.sqrt(3) / 2) * parent.side_length
            
            if parent.rotate_base_down:
                # For case 6: Triangle with base down
                return (0, -h/3 + shape.radius)
            else:
                # For case 4: Triangle with base up
                if isinstance(parent.parent, Square):
                    S: float = parent.parent.side_length / 2
                    triangle_bottom_y: float = S - h
                    return (0, triangle_bottom_y + shape.radius)
                return (0, 0)
        elif isinstance(parent, Square):
            if isinstance(parent.parent, Triangle):
                h: float = (np.sqrt(3) / 2) * parent.parent.side_length
                return (0, -h/3 + parent.side_length/2)
            return (0, 0)
        return (0, 0)

    def get_vertices(self, shape: 'Circle', 
                    parent: Optional[Shape] = None,
                    grandparent: Optional[Shape] = None) -> List[Tuple[float, float]]:
        pos: Tuple[float, float] = self.get_position(shape, parent, grandparent)
        angles: np.ndarray = np.linspace(0, 360, 9)[:-1]  # 0 to 315 degrees
        return [(shape.radius * np.cos(np.deg2rad(a)) + pos[0],
                shape.radius * np.sin(np.deg2rad(a)) + pos[1])
                for a in angles]

class SquarePositioner(ShapePositioner):
    def get_position(self, shape: 'Square', 
                    parent: Optional[Shape] = None,
                    grandparent: Optional[Shape] = None) -> Tuple[float, float]:
        if isinstance(parent, Triangle):
            h: float = (np.sqrt(3) / 2) * parent.side_length
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

    def get_vertices(self, shape: 'Square', 
                    parent: Optional[Shape] = None,
                    grandparent: Optional[Shape] = None) -> List[Tuple[float, float]]:
        pos = self.get_position(shape, parent, grandparent)
        if isinstance(parent, Triangle):
            h: float = (np.sqrt(3) / 2) * parent.side_length
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
    def get_position(self, shape: 'Triangle', 
                    parent: Optional[Shape] = None,
                    grandparent: Optional[Shape] = None) -> Tuple[float, float]:
        return (0, 0)

    def get_vertices(self, shape: 'Triangle', 
                    parent: Optional[Shape] = None,
                    grandparent: Optional[Shape] = None) -> List[Tuple[float, float]]:
        pos = self.get_position(shape, parent, grandparent)
        h: float = (np.sqrt(3) / 2) * shape.side_length

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
    def __init__(self, radius: float, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.radius: float = radius
        self._positioner: CirclePositioner = CirclePositioner()

    def plot(self, ax: Axes) -> Patch:
        pos: Tuple[float, float] = self._positioner.get_position(
            self, 
            self.parent,
            self.parent.parent if self.parent else None
        )
        circle: MplCircle = MplCircle(
            pos, self.radius,
            edgecolor=self.color,
            facecolor=self.facecolor,
            linewidth=self.linewidth,
            alpha=self.alpha
        )
        ax.add_patch(circle)
        return circle

    def get_vertices(self) -> List[Tuple[float, float]]:
        return self._positioner.get_vertices(self, self.parent,
                                           self.parent.parent if self.parent else None)

# Square Shape
class Square(Shape):
    def __init__(self, side_length: float, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.side_length: float = side_length
        self._positioner: SquarePositioner = SquarePositioner()

    def plot(self, ax: Axes) -> Patch:
        vertices = self._positioner.get_vertices(self, self.parent,
                                               self.parent.parent if self.parent else None)
        square = MplPolygon(vertices, closed=True,
                           edgecolor=self.color,
                           facecolor=self.facecolor,
                           linewidth=self.linewidth,
                           alpha=self.alpha)
        ax.add_patch(square)
        return square

    def get_vertices(self) -> List[Tuple[float, float]]:
        return self._positioner.get_vertices(self, self.parent,
                                           self.parent.parent if self.parent else None)

# Triangle Shape
class Triangle(Shape):
    def __init__(self, side_length: float, rotate_base_down: bool = False, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.side_length: float = side_length
        self.rotate_base_down: bool = rotate_base_down
        self._positioner: TrianglePositioner = TrianglePositioner()

    def plot(self, ax: Axes) -> Patch:
        vertices = self._positioner.get_vertices(self, self.parent,
                                               self.parent.parent if self.parent else None)
        triangle = MplPolygon(vertices, closed=True,
                            edgecolor=self.color,
                            facecolor=self.facecolor,
                            linewidth=self.linewidth,
                            alpha=self.alpha)
        ax.add_patch(triangle)
        return triangle

    def get_vertices(self) -> List[Tuple[float, float]]:
        return self._positioner.get_vertices(self, self.parent,
                                           self.parent.parent if self.parent else None)

# ShapeHierarchy class to manage the hierarchy and plotting
class ShapeHierarchy:
    def __init__(self, hierarchy: List[str], subplot_pos: Tuple[int, int, int], 
                 fig: Figure, colors: Optional[Dict[str, str]] = None, 
                 alphas: Optional[Dict[str, float]] = None) -> None:
        """
        Initialize the hierarchy with a list of shapes from outermost to innermost.

        Parameters:
        - hierarchy: List of shapes in order
        - subplot_pos: Tuple of (row, col, index) for subplot positioning
        - fig: Matplotlib figure object
        - colors: Dict mapping shapes to colors
        - alphas: Dict mapping shapes to alpha transparency
        """
        self.hierarchy: List[str] = hierarchy
        self.colors: Dict[str, str] = colors or {
            'circle': 'green',
            'square': 'red',
            'triangle': 'blue'
        }
        self.alphas: Dict[str, float] = alphas or {
            'circle': 0.5,
            'square': 0.5,
            'triangle': 0.5
        }
        self.shape_sizes: Dict[str, float] = {}
        self.fig: Figure = fig
        self.ax: Axes = fig.add_subplot(*subplot_pos)
        self.ax.set_aspect('equal')
        self.initial_size: float = 1.0  # This will be set from main()

    # Inscribing functions
    def inscribe_square_in_circle(self, radius: float) -> float:
        return radius * np.sqrt(2)

    def inscribe_circle_in_square(self, side_length: float) -> float:
        return side_length / 2

    def inscribe_triangle_in_square(self, side_length: float) -> float:
        """
        Calculate the side length of an equilateral triangle inscribed in a square,
        such that it touches the square at the middle of its left and right sides.
        The triangle's side length should equal the square's side length.
        """
        return side_length

    def inscribe_square_in_triangle(self, side_length: float) -> float:
        """
        Calculates the side length of a square inscribed in an equilateral triangle.
        Formula: S = (sqrt(3) * L) / (2 + sqrt(3))
        """
        return (np.sqrt(3) * side_length) / (2 + np.sqrt(3))

    def inscribe_triangle_in_circle(self, radius: float) -> float:
        """
        Calculates the side length of an equilateral triangle inscribed in a circle.
        Formula: T = 2 * R * sin(60°) = R * sqrt(3)
        """
        return 2 * radius * np.sin(np.pi / 3)  # 2R sin(60°) = R * sqrt(3)

    def inscribe_circle_in_triangle(self, side_length: float) -> float:
        """
        Calculates the radius of a circle inscribed in an equilateral triangle.
        Formula: r = (a * sqrt(3)) / 6
        """
        return (side_length * np.sqrt(3)) / 6  # r = (a√3)/6

    def calculate_inscribed_size(self, outer_shape: str, inner_shape: str, outer_size: float) -> float:
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

    def plot(self) -> None:
        """
        Plot the hierarchy of shapes.
        """
        if not self.hierarchy:
            raise ValueError("Hierarchy list is empty.")

        # Calculate vertical offset if triangle is the outer shape
        y_offset: float = 0
        if self.hierarchy[0] == 'triangle':
            h: float = (np.sqrt(3) / 2) * self.initial_size
            y_offset = -h/3

        # Create transform for the offset
        offset_transform: transforms.Transform = self.ax.transData + transforms.ScaledTranslation(0, y_offset, self.fig.dpi_scale_trans)
        
        # Plot outermost shape
        outer_shape: str = self.hierarchy[0]
        shape_obj: Shape = self.create_shape(outer_shape, self.initial_size)
        patch: Patch = shape_obj.plot(self.ax)
        patch.set_transform(offset_transform)
        self.shape_sizes[outer_shape] = self.initial_size
        previous_shape: Shape = shape_obj

        # Plot subsequent shapes
        for i in range(1, len(self.hierarchy)):
            outer: str = self.hierarchy[i - 1]
            inner: str = self.hierarchy[i]
            outer_size: float = self.shape_sizes[outer]
            inner_size: float = self.calculate_inscribed_size(outer, inner, outer_size)
            self.shape_sizes[inner] = inner_size
            shape_obj = self.create_shape(inner, inner_size)
            shape_obj.set_parent(previous_shape)  # Set parent reference
            patch = shape_obj.plot(self.ax)
            patch.set_transform(offset_transform)
            previous_shape = shape_obj

        # Use consistent limits for all plots
        limit: float = 1.1
        self.ax.set_xlim(-limit, limit)
        self.ax.set_ylim(-limit, limit)
        self.ax.axis('off')

    def create_shape(self, shape_type: str, size: float) -> Shape:
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
            rotate_base_down: bool = (shape_type == self.hierarchy[0])
            return Triangle(side_length=size, 
                      rotate_base_down=rotate_base_down,
                      color=self.colors['triangle'], 
                      alpha=self.alphas['triangle'])
        else:
            raise ValueError(f"Unsupported shape type: {shape_type}")

# Helper functions for point-in-shape tests
def point_in_circle(x: float, y: float, R: float, center: Tuple[float, float] = (0, 0)) -> bool:
    """Check if point (x,y) is inside a circle with radius R and given center."""
    x_rel = x - center[0]
    y_rel = y - center[1]
    return (x_rel**2 + y_rel**2) <= R**2 + 1e-8

def point_in_square(x: float, y: float, S: float, center: Tuple[float, float] = (0, 0)) -> bool:
    """Check if point (x,y) is inside a square with side length S and given center."""
    x_rel = x - center[0]
    y_rel = y - center[1]
    return (abs(x_rel) <= S/2 + 1e-8) and (abs(y_rel) <= S/2 + 1e-8)

def point_in_triangle(x: float, y: float, T: float, rotate_base_down: bool = False, 
                     center: Tuple[float, float] = (0, 0)) -> bool:
    """
    Check if point (x,y) is inside an equilateral triangle with side length T.
    
    Args:
        x, y: Point coordinates
        T: Triangle side length
        rotate_base_down: If True, triangle has base at bottom, otherwise base at top
        center: Center point of the triangle
    """
    # Translate point to origin-centered coordinate system
    x_rel = x - center[0]
    y_rel = y - center[1]
    
    h: float = (np.sqrt(3) / 2) * T
    
    if rotate_base_down:
        # Triangle with base at bottom
        A: np.ndarray = np.array([-T/2, -h/3])  # Bottom left
        B: np.ndarray = np.array([T/2, -h/3])   # Bottom right
        C: np.ndarray = np.array([0, 2*h/3])    # Top
    else:
        # Triangle with base at top
        A: np.ndarray = np.array([-T/2, h/2])   # Top left
        B: np.ndarray = np.array([T/2, h/2])    # Top right
        C: np.ndarray = np.array([0, -h/2])     # Bottom
    
    # Compute vectors
    v0: np.ndarray = C - A
    v1: np.ndarray = B - A
    v2: np.ndarray = np.array([x_rel, y_rel]) - A

    # Compute dot products
    dot00: float = np.dot(v0, v0)
    dot01: float = np.dot(v0, v1)
    dot02: float = np.dot(v0, v2)
    dot11: float = np.dot(v1, v1)
    dot12: float = np.dot(v1, v2)

    # Compute barycentric coordinates
    denom: float = dot00 * dot11 - dot01 * dot01
    if abs(denom) < 1e-8:  # Avoid division by zero
        return False

    u: float = (dot11 * dot02 - dot01 * dot12) / denom
    v: float = (dot00 * dot12 - dot01 * dot02) / denom

    # Check if point is in triangle
    return (u >= -1e-8) and (v >= -1e-8) and (u + v <= 1 + 1e-8)

# Unit Tests
class TestShapeHierarchy(unittest.TestCase):
    def setUp(self) -> None:
        # Create a figure for testing
        self.fig = plt.figure()
        # Use a simple subplot position (1,1,1) for testing
        self.subplot_pos = (1,1,1)
        self.hierarchy_manager = ShapeHierarchy(
            ['circle', 'square', 'triangle'],
            self.subplot_pos,
            self.fig
        )

    def tearDown(self) -> None:
        plt.close(self.fig)

    def test_inscribe_square_in_circle(self) -> None:
        radius = 1.0
        expected_side = np.sqrt(2)
        calculated_side = self.hierarchy_manager.inscribe_square_in_circle(radius)
        self.assertAlmostEqual(calculated_side, expected_side, places=5, 
                             msg="Square inscribed in circle incorrect.")

    def test_inscribe_circle_in_square(self) -> None:
        side = np.sqrt(2)
        expected_radius = 0.70710678118  # sqrt(2)/2
        calculated_radius = self.hierarchy_manager.inscribe_circle_in_square(side)
        self.assertAlmostEqual(calculated_radius, expected_radius, places=5, 
                             msg="Circle inscribed in square incorrect.")

    def test_inscribe_triangle_in_square(self) -> None:
        side = 1.0  # Square's side_length
        expected_triangle_side = side  # Triangle side equals square side
        calculated_triangle_side = self.hierarchy_manager.inscribe_triangle_in_square(side)
        self.assertAlmostEqual(calculated_triangle_side, expected_triangle_side, places=5, 
                             msg="Triangle inscribed in square incorrect.")

    def test_inscribe_circle_in_triangle(self) -> None:
        side = np.sqrt(2)
        expected_radius = (side * np.sqrt(3)) / 6  # (sqrt(2)*sqrt(3))/6 = sqrt(6)/6 ≈0.4082
        calculated_radius = self.hierarchy_manager.inscribe_circle_in_triangle(side)
        self.assertAlmostEqual(calculated_radius, expected_radius, places=5, 
                             msg="Circle inscribed in triangle incorrect.")

    def test_inscribe_triangle_in_circle(self) -> None:
        radius = 1.0
        expected_side = np.sqrt(3)  # 2*sin(60)*1 = sqrt(3)≈1.7320
        calculated_side = self.hierarchy_manager.inscribe_triangle_in_circle(radius)
        self.assertAlmostEqual(calculated_side, expected_side, places=5, 
                             msg="Triangle inscribed in circle incorrect.")

    def test_inscribe_square_in_triangle(self) -> None:
        side = 1.0
        expected_side = (np.sqrt(3) * side) / (2 + np.sqrt(3))  # ≈0.464
        calculated_side = self.hierarchy_manager.inscribe_square_in_triangle(side)
        self.assertAlmostEqual(calculated_side, expected_side, places=5, 
                             msg="Square inscribed in triangle incorrect.")

    def test_invalid_inscribing_relationship(self):
        with self.assertRaises(ValueError):
            self.hierarchy_manager.calculate_inscribed_size('circle', 'circle', 1.0)

    def test_invalid_shape_type(self):
        with self.assertRaises(ValueError):
            self.hierarchy_manager.create_shape('hexagon', 1.0)

    def test_alignment_C_S_T(self):
        # Hierarchy: Circle > Square > Triangle
        hierarchy = ['circle', 'square', 'triangle']
        fig = plt.figure()
        try:
            shape_hierarchy = ShapeHierarchy(hierarchy, (1,1,1), fig)
            R = 1.0
            S = shape_hierarchy.inscribe_square_in_circle(R)
            T = shape_hierarchy.inscribe_triangle_in_square(S)

            # Get Triangle vertices
            triangle = Triangle(T)
            vertices = triangle.get_vertices()

            # Check each vertex is within Square
            for x, y in vertices:
                self.assertTrue(point_in_square(x, y, S), 
                              f"Triangle vertex ({x}, {y}) not inside Square.")

            # Check Square vertices are within Circle
            square = Square(S)
            square_vertices = square.get_vertices()
            for x, y in square_vertices:
                self.assertTrue(point_in_circle(x, y, R), 
                              f"Square vertex ({x}, {y}) not inside Circle.")
        finally:
            plt.close(fig)

    def test_alignment_C_T_S(self):
        hierarchy = ['circle', 'triangle', 'square']
        fig = plt.figure()
        try:
            shape_hierarchy = ShapeHierarchy(hierarchy, (1,1,1), fig)
            R = 1.0
            T = shape_hierarchy.inscribe_triangle_in_circle(R)
            S = shape_hierarchy.inscribe_square_in_triangle(T)

            # Create shapes with proper parent relationships
            circle = Circle(R)
            triangle = Triangle(T)
            square = Square(S)
            
            triangle.set_parent(circle)
            square.set_parent(triangle)

            # Get vertices and positions
            square_vertices = square.get_vertices()
            triangle_vertices = triangle.get_vertices()
            
            # Check each square vertex is within the triangle
            for x, y in square_vertices:
                self.assertTrue(point_in_triangle(x, y, T), 
                              f"Square vertex ({x}, {y}) not inside Triangle.")

            # Check triangle vertices are within the circle
            for x, y in triangle_vertices:
                self.assertTrue(point_in_circle(x, y, R), 
                              f"Triangle vertex ({x}, {y}) not inside Circle.")
        finally:
            plt.close(fig)

    def test_alignment_S_T_C(self):
        hierarchy = ['square', 'triangle', 'circle']
        fig = plt.figure()
        try:
            shape_hierarchy = ShapeHierarchy(hierarchy, (1,1,1), fig)
            S = 1.0
            T = shape_hierarchy.inscribe_triangle_in_square(S)
            R = shape_hierarchy.inscribe_circle_in_triangle(T)

            # Create shapes with proper parent relationships
            square = Square(S)
            triangle = Triangle(T)
            circle = Circle(R)
            
            triangle.set_parent(square)
            circle.set_parent(triangle)

            # Get vertices and positions
            circle_vertices = circle.get_vertices()
            triangle_vertices = triangle.get_vertices()
            
            # Check each circle vertex is within the triangle
            for x, y in circle_vertices:
                self.assertTrue(point_in_triangle(x, y, T), 
                              f"Circle vertex ({x}, {y}) not inside Triangle.")

            # Check triangle vertices are within the square
            for x, y in triangle_vertices:
                self.assertTrue(point_in_square(x, y, S), 
                              f"Triangle vertex ({x}, {y}) not inside Square.")
        finally:
            plt.close(fig)

    def test_alignment_T_S_C(self):
        hierarchy = ['triangle', 'square', 'circle']
        fig = plt.figure()
        try:
            shape_hierarchy = ShapeHierarchy(hierarchy, (1,1,1), fig)
            T = 1.0
            S = shape_hierarchy.inscribe_square_in_triangle(T)
            R = shape_hierarchy.inscribe_circle_in_square(S)

            # Create shapes with proper parent relationships
            triangle = Triangle(T, rotate_base_down=True)
            square = Square(S)
            circle = Circle(R)
            
            square.set_parent(triangle)
            circle.set_parent(square)

            # Get vertices and positions using the positioner system
            circle_vertices = circle.get_vertices()
            square_pos = square._positioner.get_position(square, triangle)
            circle_pos = circle._positioner.get_position(circle, square)

            # Check each vertex is within parent shape
            for x, y in circle_vertices:
                self.assertTrue(point_in_square(x, y, S, center=square_pos), 
                              f"Circle vertex ({x}, {y}) not inside Square.")

            square_vertices = square.get_vertices()
            for x, y in square_vertices:
                self.assertTrue(point_in_triangle(x, y, T, rotate_base_down=True), 
                              f"Square vertex ({x}, {y}) not inside Triangle.")
        finally:
            plt.close(fig)
    def check_vertices_in_shape(self, vertices: List[Tuple[float, float]], 
                                container_shape: Shape, 
                                shape_name: str,
                                container_name: str) -> None:
        """Helper method to check if vertices are inside a containing shape."""
        for x, y in vertices:
            if isinstance(container_shape, Circle):
                inside = point_in_circle(x, y, container_shape.radius)
            elif isinstance(container_shape, Square):
                inside = point_in_square(x, y, container_shape.side_length)
            elif isinstance(container_shape, Triangle):
                inside = point_in_triangle(x, y, container_shape.side_length, 
                                        container_shape.rotate_base_down)
            else:
                raise ValueError(f"Unknown container shape type: {type(container_shape)}")
                
            self.assertTrue(inside, 
                          f"{shape_name} vertex ({x}, {y}) not inside {container_name}.")

# Main execution
def main() -> None:
    """
    Main function to execute the script.
    """
    all_shapes: List[str] = ['circle', 'square', 'triangle']
    possible_hierarchies: List[Tuple[str, ...]] = list(permutations(all_shapes))
    
    n_plots: int = len(possible_hierarchies)
    n_cols: int = 3
    n_rows: int = (n_plots + n_cols - 1) // n_cols
    
    fig: Figure = plt.figure(figsize=(15, 5 * n_rows))
    
    custom_colors: Dict[str, str] = {
        'circle': 'green',
        'square': 'red',
        'triangle': 'blue'
    }
    custom_alphas: Dict[str, float] = {
        'circle': 0.5,
        'square': 0.5,
        'triangle': 0.5
    }

    # Scale factor to make shapes fit better in the canvas
    scale: float = 0.8  # Adjust this value to make shapes larger or smaller
    
    # Base sizes scaled down to fit better
    initial_sizes: Dict[str, float] = {
        'circle': 1.0 * scale,  # radius
        'square': 2.0 * scale,  # side length
        'triangle': (2.0 * 2/np.sqrt(3)) * scale  # side length
    }

    for idx, hierarchy in enumerate(possible_hierarchies, 1):
        subplot_pos: Tuple[int, int, int] = (n_rows, n_cols, idx)
        shape_hierarchy: ShapeHierarchy = ShapeHierarchy(
            hierarchy, 
            subplot_pos, 
            fig,
            colors=custom_colors, 
            alphas=custom_alphas
        )
        
        outer_shape: str = hierarchy[0]
        shape_hierarchy.initial_size = initial_sizes[outer_shape]
        
        shape_hierarchy.plot()
        hierarchy_title: str = " > ".join([s.capitalize() for s in hierarchy])
        shape_hierarchy.ax.set_title(f"{idx}. {hierarchy_title}", fontsize=10)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Run unit tests
    print("Running unit tests...")
    unittest.main(argv=[''], exit=False)

    # Execute main plotting
    main()