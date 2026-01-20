#!/bin/bash

# ---------------- USER SETTINGS ----------------
FLOWUNSTEADY_ROOT="/home/dysco/FLOWUnsteady"
JULIA_SCRIPT="VarFLEXI-rVPM-Fenics/fluid-rvpm/fluid-vlm.jl"
PYTHON_SCRIPT="VarFLEXI-rVPM-Fenics/fluid-rvpm/force_extract.py"
# -----------------------------------------------

echo "=============================================="
echo " Running FLOWUnsteady project + force extractor"
echo "=============================================="

cd "$FLOWUNSTEADY_ROOT" || exit 1

# Start Python extractor (filesystem-based, independent of Julia env)
echo "[INFO] Starting Python force extractor..."
python3 "$PYTHON_SCRIPT" &
PYTHON_PID=$!

# Give Python time to initialize
sleep 2

# Run Julia using the FLOWUnsteady project
echo "[INFO] Starting Julia solver with FLOWUnsteady project..."
julia --project="$FLOWUNSTEADY_ROOT" "$JULIA_SCRIPT"

# Cleanup
echo "[INFO] Julia finished. Stopping Python extractor..."
kill $PYTHON_PID 2>/dev/null

echo "=============================================="
echo " Simulation finished cleanly"
echo "=============================================="
