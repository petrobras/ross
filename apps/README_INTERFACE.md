# ROSS Graphical Interface

This is a web-based graphical interface for the **ROSS** library, designed to simplify the modeling and analysis of rotating machinery. The project integrates an interactive JavaScript frontend with a robust Python (Flask) backend, enabling rotor dynamics simulations through a visual and intuitive workflow.

## 🚀 Features

- **Comprehensive Modeling:** Add and edit materials, shaft elements, disks, gears, bearings, seals, couplings, and point masses.
- **Real-Time Visualization:** 3D visualization of the rotor model that updates as elements are added.
- **Advanced Analysis Dashboards:**
    - Campbell Diagram;
    - Unbalanced Critical Speed (UCS);
    - Frequency Response;
    - Time Response;
    - Vibration Modes (2D and 3D);
    - Unbalance Response;
    - Static Analysis.
- **Python Script Generation:** Automatically export your model and analysis settings into a ready-to-run Python script using native ROSS syntax.
- **Data Portability:** Save and load your rotor models and analysis configurations using JSON files.

## 🛠️ Technologies

- **Frontend:** HTML5, CSS3, JavaScript (using Plotly.js for charting and Sortable.js for list management).
- **Backend:** Python 3, Flask (Web Server), ROSS-rotordynamics (Calculation Engine).

## 📂 Project Structure

- `index.html`: The main UI structure, defining the modeling and analysis screens.

- `style.css`: Contains all visual styling, layout definitions, and responsive design rules.

- `app.js`: Handles frontend logic, including rotor data management, dynamic form generation, and API communication with the Flask server.

- `app.py`: The Flask backend that interacts with the ROSS library to perform calculations and generate plot data.

- `requirements.txt`: List of Python dependencies required to run the server.

## 📋 Prerequisites

Ensure you have Python 3.x installed on your machine. It is highly recommended to use a virtual environment for dependency management.

## 🔧 Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/ross.git](https://github.com/your-username/ross-interface.git)
   cd ross-interface

2. **Create a virtual environment (optional):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\\Scripts\\activate

3. **Install dependencies (use the requirements.txt file):**
    ```bash
    pip install -r requirements.txt

## 💻 How to Execute

1. **Start the Backend Server:**
    Run the Flask server from your terminal
    ```bash
    python .\apps\app.py

2. **Access the Interface:**
    `app.py` automatically tries to open `frontend/index.html` in your default web browser after a short delay. If it does not open, open `apps/frontend/index.html` manually in Chrome or Firefox.

## 📦 Windows standalone bundle (PyInstaller)

Use Python **3.10+** and install dependencies from the repository root:

```bash
pip install -e .
pip install -r apps/requirements.txt pyinstaller
pyinstaller apps/ross_interface.spec --noconfirm
```

This produces `dist/ROSS-Interface/` with `ROSS-Interface.exe` and bundled assets. Zip that folder for distribution, or rely on [GitHub Actions](https://github.com/petrobras/ross/actions/workflows/windows-interface-and-docs.yml), which builds the same bundle on `windows-latest` and attaches it to the documentation site.