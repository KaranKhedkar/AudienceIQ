@echo off
title AudienceIQ - Power BI Desktop Launcher
cls
echo ====================================================================
echo             AUDIENCEIQ BUSINESS INTELLIGENCE PLATFORM
echo ====================================================================
echo.
echo Launching Microsoft Power BI Desktop...
start "" "C:\Program Files\Microsoft Power BI Desktop\bin\PBIDesktop.exe"

echo Opening folder with the 5 Staging CSV tables...
start explorer "%~dp0dashboard\powerbi"

echo.
echo ====================================================================
echo  HOW TO IMPORT DATA IN POWER BI (3 Quick Steps):
echo ====================================================================
echo  1. In Power BI Desktop, click "Get Data" -> "Text/CSV"
echo  2. Select any (or all) of the 5 CSV files from the folder:
echo       - bi_executive_kpis.csv       (High-level summary metrics)
echo       - bi_customer_segments.csv    (206K customer persona profiles)
echo       - bi_product_intelligence.csv (49K grocery products & categories)
echo       - bi_demand_forecasts.csv     (30-day forward demand projections)
echo       - bi_cross_sell_matrix.csv    (Product co-purchase pairs)
echo  3. Click "Load" to start visualizing!
echo ====================================================================
echo.
pause
