@echo off
REM ===========================================================================
REM  RUN_ALL.bat — Master runner for E-Commerce Data Warehouse & Mining Platform
REM  Patterned on the Enterprise Route Simulator DWM Pipeline Architecture
REM ===========================================================================

echo.
echo ===================================================================
echo   E-COMMERCE DATA WAREHOUSE ^& DATA MINING — MASTER SETUP RUNNER
echo ===================================================================
echo.

set PYTHONIOENCODING=utf-8

REM ── 1. Install dependencies ──────────────────────────────────────────────
echo [STEP 1/7] Installing / Verifying Python dependencies...
python -m pip install -r requirements.txt --quiet
if %ERRORLEVEL% neq 0 (echo ERROR: pip install failed & pause & exit /b 1)
echo   Done.

REM ── 2. Audit current DB state ────────────────────────────────────────────
echo.
echo [STEP 2/7] Auditing database and warehouse row counts...
python audit.py

REM ── 3. Load warehouse dimensions and facts ───────────────────────────────
echo.
echo [STEP 3/7] Loading conformed Star Schema (Dimensions, Facts, Views)...
python run_warehouse.py
if %ERRORLEVEL% neq 0 (
    echo WARNING: Warehouse load had warnings. Check output above.
)

REM ── 4. Run ML suite ──────────────────────────────────────────────────────
echo.
echo [STEP 4/7] Running Machine Learning suite (Clustering, Apriori, Churn, CLV)...
python -m ml.run_all_ml
if %ERRORLEVEL% neq 0 (echo WARNING: Some ML modules failed. Check output.)

REM ── 5. Run demo simulation ───────────────────────────────────────────────
echo.
echo [STEP 5/7] Running E-Commerce Campaign & Pricing Simulator...
python simulator.py
if %ERRORLEVEL% neq 0 (echo WARNING: Simulator demo failed.)

REM ── 6. Run domain analyses ───────────────────────────────────────────────
echo.
echo [STEP 6/7] Running Customer & Product Domain Analyses...
python customer_analysis.py
python product_analysis.py

REM ── 7. Run full test suite ───────────────────────────────────────────────
echo.
echo [STEP 7/7] Running full automated verification test suite...
python tests\test_all.py
if %ERRORLEVEL% neq 0 (
    echo.
    echo SOME TESTS FAILED — Check output above.
)

REM ── Start dashboard ──────────────────────────────────────────────────────
echo.
echo ===================================================================
echo   SETUP COMPLETE — Starting Interactive Streamlit Dashboard
echo   Open: http://localhost:8501
echo   Press Ctrl+C to stop
echo ===================================================================
echo.
python -m streamlit run app.py --server.port 8501

pause
