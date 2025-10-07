Got it — here’s a **professional, clean, emoji-free README** version for your package:

---

# pyobjtools

**pyobjtools** is a lightweight Python package for loading, converting, and manipulating 3D **Wavefront OBJ files**.
It provides an easy interface to read `.obj` files, convert them to CSV format, and modify vertex data programmatically.

---

## Installation

You can install the package locally using `pip`:

```bash
pip install .
```

or, if you are developing it:

```bash
pip install -e .
```

After installation, you can import it as follows:

```python
from pyobjtools import ObjFile
```

---

## Usage Example

```python
from pyobjtools import ObjFile

# Load a 3D object (OBJ or CSV format)
obj = ObjFile("samples/cube.obj")

# Convert it to CSV format
obj.convert_to_csv("outputs")

# Remove specific vertices by their IDs
obj.remove_vertices_list_by_ids([1, 5, 10])

# Save the modified object to a new .obj file
obj.write_obj_file("outputs/cube_modified.obj")
```

---

## Features

* **Load OBJ files**: Reads `.obj` files and extracts geometry and metadata.
* **Convert to CSV**: Converts 3D models into CSV format for analysis or further processing.
* **Modify geometry**: Supports vertex removal by ID.
* **Export to OBJ**: Writes modified geometry back to `.obj` format.
* **CSV round-trip**: Allows reloading data from CSV files and converting back to OBJ format.

---

## Package Structure

```
pyobjtools/
│
├── __init__.py          # Package initializer, exports ObjFile
├── objfile.py           # Main interface class
├── csv_utils.py         # Handles CSV read/write
├── io_utils.py          # Handles OBJ file I/O
└── geometry_ops.py      # Handles geometry operations
```

---

## Class Overview

### `class ObjFile(obj_file_path: str)`

Represents a 3D object file and provides methods for conversion and manipulation.

#### Methods

| Method                             | Description                                |
| ---------------------------------- | ------------------------------------------ |
| `convert_to_csv(output_path)`      | Converts the current object to CSV format. |
| `write_obj_file(write_path)`       | Writes the object back to an `.obj` file.  |
| `remove_vertices_list_by_ids(ids)` | Removes vertices with the given IDs.       |

---

## About the Wavefront OBJ Format

The Wavefront OBJ format was created by **Wavefront Technologies** in the 1980s as a simple text-based standard for storing 3D geometry.
The `.obj` extension stands for **“object file”**, and it remains one of the most widely supported 3D model formats today.

---

## License

MIT License © 2025 Mostafa Osman

---
