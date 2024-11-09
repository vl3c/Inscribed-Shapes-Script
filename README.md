# Inscribed Shapes AI Experiment

This project is an experiment to test the capabilities of advanced AI code generation and debugging tools (specifically Cursor) when dealing with geometric problems. The focus is on drawing inscribed shapes - a task that typically challenges AI systems.

## Background

Advanced AI models often struggle with precise geometric calculations and visualizations, particularly when it comes to:
- Correctly inscribing shapes within other shapes
- Maintaining proper proportions and alignments
- Handling different geometric cases and orientations

This script serves as a test case for these challenges, implementing six different cases of inscribed shapes:
1. Square > Triangle > Circle
2. Circle > Triangle > Square
3. Triangle > Circle > Square
4. Square > Triangle > Circle
5. Circle > Square > Triangle
6. Triangle > Square > Circle

## Requirements

- Python 3.x
- matplotlib
- numpy

## Installation

**Create virtual environment**
python -m venv venv

**Activate virtual environment**
**On Windows:**
.\venv\Scripts\activate
**On Unix or MacOS:**
source venv/bin/activate

**Install dependencies**
pip install -r requirements.txt

## Usage

Run the script to see the visualization of all six inscribed shape cases:
python inscribed_shapes.py

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.