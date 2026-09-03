@echo off
title AudienceIQ - Power BI Desktop Launcher
cls
echo ====================================================================
echo             AUDIENCEIQ BUSINESS INTELLIGENCE PLATFORM
echo ====================================================================
echo.
echo Launching Microsoft Power BI Desktop...
start "" "C:\Program Files\Microsoft Power BI Desktop\bin\PBIDesktop.exe"

echo Opening staging data folder...
start explorer "%~dp0"

echo.
echo ====================================================================
echo  HOW TO IMPORT DATA IN POWER BI (3 Quick Steps):
echo ====================================================================
echo  1. In Power BI Desktop, click "Get Data" -> "Text/CSV"
echo  2. Select any of the 5 staging CSV files in this directory:
echo       - bi_executive_kpis.csv
echo       - bi_customer_segments.csv
echo       - bi_product_intelligence.csv
echo       - bi_demand_forecasts.csv
echo       - bi_cross_sell_matrix.csv
echo  3. Click "Load"
echo ====================================================================
echo.
pause
