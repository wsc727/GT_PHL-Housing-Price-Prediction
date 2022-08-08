"""
Please be sure to run this in the appropriate Python environment so you have all the right libraries.
Run this python script to download four different datasets.
Author: Raghuveer Krishnamurthy (raghuveer3)
"""

import pandas as pd
import numpy as np
import requests
import fastparquet as fp
import time

phl = True # Download Philadelphia data? Set to True - data from Philadelphia appears to be complete as of 11/2020
fairfax = True # Download Fairfax county data?  Set to True - data from Fairfax County, VA appears to be complete as of 11/2020
hartford = False # Download Hartford data? Currently set to False because of incomplete data provided by City of Hartford
sanfrancisco = False # Download SF data? Currently set to False because of incomplete data provided by City and County of San Francisco

col_names = dict({('PARID', 'Parcel_ID'), 
                  ('parcel_number', 'Parcel_ID'), 
                  ('STYLE_DESC', 'Building_Style'), 
                  ('category_code_description', 'Building_Category'),
                  ('YRBLT', 'Year_Built'), 
                  ('year_built', 'Year_Built'), 
                  ('number_of_rooms', 'Total_Room_Count'),
                  ('YRREMOD', 'Year_Remodeled'), 
                  ('number_stories', 'Number_of_Stories'),
                  ('RMBED', 'Bedroom_Count'), 
                  ('number_of_bathrooms', 'Full_Bathroom_Count'),
                  ('number_of_bedrooms', 'Bedroom_Count'),
                  ('FIXBATH', 'Full_Bathroom_Count'), 
                  ('FIXHALF', 'Half_Bathroom_count'), 
                  ('RECROMAREA', 'Basement_Size'), 
                  ('WBFP_PF', 'Fireplaces_Count'), 
                  ('fireplaces', 'Fireplaces_Count'), 
                  ('BSMTCAR', 'Basement_Garage_Parking_Space_Count'), 
                  ('garage_spaces', 'Basement_Garage_Parking_Space_Count'),
                  ('USER10', 'Basement_Room_Count'),
                  ('GRADE_DESC', 'Construction_Quality'), 
                  ('quality_grade', 'Construction_Quality'),
                  ('SFLA', 'Non_Basement_Area'), 
                  ('total_livable_area', 'Non_Basement_Area'),
                  ('total_area', 'Lot_Size'),
                  ('SQFT', 'Lot_Size'),
                  ('CDU_DESC', 'Physical_Condition'), 
                  ('exterior_condition', 'Physical_Condition'),                   
                  ('HEAT_DESC', 'Air_Conditioning'),
                  ('central_air', 'Air_Conditioning'),
                  ('TAXYR', 'Tax_Year'),
                  ('SALEDT', 'Sale_Date'),
                  ('sale_date', 'Sale_Date'),                  
                  ('PRICE', 'Sale_Price'),
                  ('sale_price', 'Sale_Price'),
                  ('location', 'Property_Address'),
                  ('zip_code', 'Zip_Code'),
                  ('ADRNO', 'Street_Number'),
                  ('ADRADD', 'Street_Number_Suffix'),
                  ('ADRDIR', 'Street_Direction'),
                  ('ADRSTR', 'Actual_Street_Name'),
                  ('ADRSUF', 'Street_Type'),
                  ('ADRSUF2', 'Street_Name_Suffix'),
                  ('TAXDIST', 'Tax_District'),
                  ('topography', 'Elevation_Type'), 
                  ('other_building', 'Aux_Unit'),
                  ('unit', 'Unit'),
                  ('TAXDIST_DESC', 'Tax_District_Description')})

if phl:    
    transactions = pd.DataFrame()
    assessment = pd.DataFrame()
    years = np.arange(2000, 2021)

    for year in years:
        req = str("https://phl.carto.com/api/v2/sql?q=SELECT * FROM RTT_SUMMARY WHERE document_date >= '") + str(year) + str('-01-01') + str("' AND document_date <= '") + str(year) + str('-12-31') + "'"
        propertyTransactions = requests.get(req)
        propertyTransactions = pd.DataFrame(propertyTransactions.json()['rows'])
        transactions = pd.concat([transactions, propertyTransactions], sort=False, ignore_index=True)

        req = str("https://phl.carto.com/api/v2/sql?q=SELECT * FROM opa_properties_public WHERE sale_date >= '") + str(year) + str('-01-01') + str("' AND sale_date <= '") + str(year) + str('-12-31') + "'"
        req = str("https://phl.carto.com/api/v2/sql?q=SELECT *, ST_Y(the_geom) AS lat, ST_X(the_geom) AS lng, ST_AsText(the_geom) AS the_geom_text FROM opa_properties_public WHERE sale_date >= '") + str(year) + str('-01-01') + str("' AND sale_date <= '") + str(year) + str('-12-31') + "'"
        
        propertyAssessments = requests.get(req)
        propertyAssessments = pd.DataFrame(propertyAssessments.json()['rows'])
        assessment = pd.concat([assessment, propertyAssessments], sort=False, ignore_index=True)
        
    fp.write("PHL_assessment_Raw.parq", assessment)
    fp.write("PHL_transactions_Raw.parq", transactions)
    
    # Philadelphia Cleaning
    assessment.rename(columns=col_names, inplace=True)
    assessment = assessment[assessment['category_code'].isin(['1', '2'])]
    assessment['Air_Conditioning'] = np.where(assessment['Air_Conditioning'] == 'Y', 'Central A/C', 'None')
    assessment['Sale_Date'] = pd.to_datetime(assessment['Sale_Date'], infer_datetime_format=True)
    assessment['Basement_Room_Count'] = np.where(assessment['basements'].isin(['0', None]), 0, 1)
    assessment['Fireplaces_Count'] = np.where(assessment['Fireplaces_Count'].isnull(), 0, assessment['Fireplaces_Count'])
    assessment['Physical_Condition'] = np.where(assessment['Physical_Condition'].isnull(), '3', assessment['Physical_Condition'])
    assessment['Basement_Garage_Parking_Space_Count'] = np.where(assessment['Basement_Garage_Parking_Space_Count'].isnull(), 0, assessment['Basement_Garage_Parking_Space_Count'])
    assessment['Aux_Unit'] = np.where(assessment['Aux_Unit'] == 'Y', True, False)
    assessment = assessment[~(assessment['Full_Bathroom_Count'].isnull())]
    assessment = assessment[~(assessment['Bedroom_Count'].isnull())]
    assessment = assessment[~(assessment['unfinished']=='U')]

    assessment['Elevation_Type'] = np.where(assessment['Elevation_Type'].isin(['0', None]), 'F', assessment['Elevation_Type'])
    drop_cols = ['cartodb_id', 'the_geom', 'the_geom_webmercator', 'assessment_date', 'basements', 'depth', 'date_exterior_condition', 'unfinished', 'beginning_point', 'book_and_page', 'building_code', 'building_code_description', 
                 'category_code', 'census_tract', 'cross_reference', 'frontage', 'fuel', 'utility', 'exempt_building', 'exempt_land', 'garage_type', 'general_construction', 'geographic_ward', 'homestead_exemption', 
                 'house_extension', 'house_number', 'interior_condition', 'type_heater', 'view_type', 'year_built_estimate', 'mailing_address_1', 'mailing_address_2', 'mailing_care_of', 'mailing_city_state', 'mailing_street', 
                 'mailing_zip', 'zoning', 'objectid', 'owner_1', 'owner_2', 'recording_date', 'registry_number', 'taxable_building', 'taxable_land', 'parcel_shape', 'market_value', 'market_value_date',
                 'separate_utilities', 'sewer', 'site_type', 'state_code', 'street_code', 'street_designation', 'street_direction', 'street_name', 'suffix', 'off_street_open']
    assessment.drop(labels=drop_cols, axis=1, inplace=True)
    
    fp.write("PHL_assessment_clean.parq", assessment)
    
if hartford:    
    req = str("https://data.hartford.gov/resource/uepu-9ktm.json?$limit=30000")
    assessment = requests.get(req).json()
    assessment = pd.DataFrame(propertyTransactions)
    fp.write("Hartford_assessment_raw.parq", assessment)

if sanfrancisco:
    req = str("https://data.sfgov.org/resource/wv5m-vpq2.json?$limit=3000000")
    assessment = requests.get(req).json()
    assessment = pd.DataFrame(assessment)
    fp.write('SF_assessment_raw.parq', assessment[~assessment['current_sales_date'].isnull()])

# Fairfax County, VA
if fairfax:    
    req = str("https://opendata.arcgis.com/datasets/53ee1065351c4273ab91ba2e6cfbbc6d_2.geojson")
    dwelling_data = requests.get(req).json()
    dwelling_data = pd.DataFrame.from_dict(dwelling_data['features'])
    dwelling_data = dwelling_data['properties'].tolist()
    dwelling_data = pd.DataFrame.from_records(dwelling_data)
    fp.write('Fairfax_County_dwelling_data_raw.parq', dwelling_data)
    
    req = str("https://opendata.arcgis.com/datasets/764b1798c0434003a862e2734ba2b705_1.geojson")
    salesData = requests.get(req).json()
    salesData = pd.DataFrame.from_dict(salesData['features'])
    salesData = salesData['properties'].tolist()
    salesData = pd.DataFrame.from_records(salesData)
    fp.write('Fairfax_County_sales_data_raw.parq', salesData)

    req = str("https://opendata.arcgis.com/datasets/0c3415baff124473832c0e821c0a4ddc_1.geojson")
    legalData = requests.get(req).json()
    legalData = pd.DataFrame.from_dict(legalData['features'])
    legalData = legalData['properties'].tolist()
    legalData = pd.DataFrame.from_records(legalData)
    fp.write('Fairfax_County_legal_data_raw.parq', legalData)
    
    addressData = pd.DataFrame()
    req = str("https://services1.arcgis.com/ioennV6PpG5Xodq0/arcgis/rest/services/OpenData_A2/FeatureServer/0/query?where=1%3D1&objectIds=&time=&geometry=&geometryType=esriGeometryEnvelope&inSR=&spatialRel=esriSpatialRelIntersects&resultType=none&distance=0.0&units=esriSRUnit_Meter&returnGeodetic=false&outFields=*&returnGeometry=true&featureEncoding=esriDefault&multipatchOption=xyFootprint&maxAllowableOffset=&geometryPrecision=&outSR=&datumTransformation=&applyVCSProjection=false&returnIdsOnly=false&returnUniqueIdsOnly=false&returnCountOnly=True&returnExtentOnly=false&returnQueryGeometry=false&returnDistinctValues=false&cacheHint=false&orderByFields=&groupByFieldsForStatistics=&outStatistics=&having=&resultOffset=&resultRecordCount=&returnZ=false&returnM=false&returnExceededLimitFeatures=true&quantizationParameters=&sqlFormat=none&f=pjson&token=")
    limit = requests.get(req).json()['count']
    offset = 0
    i = 0

    while limit > 0:

        req = str("https://services1.arcgis.com/ioennV6PpG5Xodq0/arcgis/rest/services/OpenData_A2/FeatureServer/0/query?where=1%3D1&objectIds=&time=&geometry=&geometryType=esriGeometryEnvelope&inSR=&spatialRel=esriSpatialRelIntersects&resultType=none&distance=0.0&units=esriSRUnit_Meter&returnGeodetic=false&outFields=*&returnGeometry=true&featureEncoding=esriDefault&multipatchOption=xyFootprint&maxAllowableOffset=&geometryPrecision=&outSR=&datumTransformation=&applyVCSProjection=false&returnIdsOnly=false&returnUniqueIdsOnly=false&returnCountOnly=false&returnExtentOnly=false&returnQueryGeometry=false&returnDistinctValues=false&cacheHint=false&orderByFields=&groupByFieldsForStatistics=&outStatistics=&having=&resultOffset=") + str(offset) + str("&resultRecordCount=&returnZ=false&returnM=false&returnExceededLimitFeatures=true&quantizationParameters=&sqlFormat=none&f=pjson&token=")
        temp_df = requests.get(req).json()
        temp_df = pd.DataFrame.from_dict(temp_df['features'])
        temp_df = pd.DataFrame.from_records(temp_df['attributes'])
        addressData = pd.concat([addressData, temp_df], axis=0, sort=False, ignore_index=True)
        offset += 1000
        limit -= 1000
        i += 1
        if i % 10 == 0:
            time.sleep(5)
            
    addressData.to_csv("Fairfax_VA_Address_Data_Raw.csv", index=False)
    fp.write("Fairfax_VA_Address_Data_Raw.parq", addressData)
    
    # Cleaning Dwelling Data
    # Fill N/As and None with 0.0 or other values as appropriate
    # Standardizing column names

    dwelling_data['FIXBATH'].fillna(value=0.0, inplace=True)
    dwelling_data['FIXHALF'].fillna(value=0.0, inplace=True)
    dwelling_data['RECROMAREA'].fillna(value=0.0, inplace=True)
    dwelling_data['WBFP_PF'].fillna(value=0.0, inplace=True)
    dwelling_data['BSMTCAR'].fillna(value=0.0, inplace=True)
    dwelling_data['USER10'].fillna(value='0', inplace=True)
    dwelling_data['GRADE_DESC'].fillna(value='Average', inplace=True)
    dwelling_data['CDU_DESC'].fillna(value='Average', inplace=True)
    dwelling_data.drop(labels=['OBJECTID', 'CreationDate', 'Creator', 'EditDate', 'Editor', 'EFFYR', 'USER6', 'BSMT_DESC', 'EXTWALL_DESC', 'USER13_DESC', 'USER7_DESC', 'USER9_DESC'], axis=1, inplace=True)
    dwelling_data.rename(columns=col_names, inplace=True)
    dwelling_data['Basement_Room_Count'] = np.where((dwelling_data['Basement_Room_Count'] == '0'), np.where(dwelling_data['Basement_Size'] > 0.0, '1', dwelling_data['Basement_Room_Count']), dwelling_data['Basement_Room_Count'])
    dwelling_data['Air_Conditioning'] = np.where(dwelling_data['Air_Conditioning'] == 'Central', 'Central A/C', dwelling_data['Air_Conditioning'])
    dwelling_data['Air_Conditioning'] = np.where(dwelling_data['Air_Conditioning'].isnull(), 'None', dwelling_data['Air_Conditioning'])

    salesData.drop(labels='OBJECTID', axis=1, inplace=True)
    salesData = salesData[salesData['PRICE']>0]
    salesData = salesData.drop_duplicates(keep='first')

    # Cleaning Sales data
    validSaleType = ['Valid and verified sale', 'Pending verification', 'Valid and verified multi-parcel sale', 'Land sale but land improved after sale', 
                     'Sale of finished bldg. imp assmt < 100%', 'In two or more assmt jurisdictions', 'Assmt includes improvements after sale', 'Transfer of partial interest',
                     'Multi-parcel sale pending verification', 'Finished lot. lot assmt part finished', 'New subdivison', 'Division of property after date of sale',
                     'Rezoned after date of sale', 'Purchased by tenant', 'Resubdivision', 'Sale from lender - valid sale price', 'Atypical financing', 
                     'Price reflects leased fee value', 'Price reflects anticipated rezoning', 'Price reflects future rezone-multi-parce', 'Multi-parcel lender sale - valid price',
                     'Finished lot. ot assmt<100%-multi-parce', 'Portfolio/Bulk Sale', 'Net Lease Sale', 'Net Lease Sale Multiple']

    salesData = salesData[salesData['SALEVAL_DESC'].isin(validSaleType)]
    salesData.rename(columns=col_names, inplace=True)
    salesData = salesData[['Parcel_ID', 'Sale_Date', 'Sale_Price']]
    salesData['Sale_Date'] = pd.to_datetime(salesData['Sale_Date'], infer_datetime_format=True)
    
    legalData = legalData[~legalData['ADRNO'].isnull()]
    legalData.drop(labels=['OBJECTID', 'CreationDate', 'Creator', 'EditDate', 'Editor'], axis=1, inplace=True)
    legalData.rename(columns=col_names, inplace=True)
    legalData = legalData[~legalData['Street_Number'].isnull()]
    legalData['Street_Number'] = legalData['Street_Number'].astype(int)
    legalData = legalData[~legalData['Actual_Street_Name'].isnull()]
    legalData = legalData[['Parcel_ID', 'Street_Number', 'Street_Number_Suffix', 'Street_Direction', 'Actual_Street_Name', 'Street_Type', 'Street_Name_Suffix', 'Tax_District', 'Tax_District_Description']]
    
    fairfax_county = salesData.merge(right=legalData, how='left', on='Parcel_ID')
    fairfax_county = fairfax_county [~fairfax_county ['Street_Number'].isnull()]
    fairfax_county['Street_Number'] = fairfax_county['Street_Number'].astype(int).astype(str)
    fairfax_county = fairfax_county.merge(right=dwelling_data, how='left', on='Parcel_ID')
    
    addressData['Actual_Address'] = addressData['ADDRESS_1'].astype(str) + np.where(addressData['UNIT_TYPE'].isnull(), str(""), str(" ") + addressData['UNIT_TYPE'].astype(str)) + np.where(addressData['UNIT_NUMBER'].isnull(), str(""), str(" ") + addressData['UNIT_NUMBER'].astype(str)) + str (", ") + addressData['CITY'].astype(str) + str(", ") + addressData['STATE'].astype(str) + str (" ") + addressData['ZIP'].astype(str)
    addressData = addressData[['ADD_LOCATION_X', 'ADD_LOCATION_Y', 'PARCEL_PIN', 'Actual_Address']]

    col_names_ffax = dict({('PARCEL_PIN', 'Parcel_ID'), 
                      ('ADD_LOCATION_X', 'lat'), 
                      ('ADD_LOCATION_Y', 'lng')})

    addressData.rename(columns=col_names_ffax, inplace=True)
    
    fairfax_county = fairfax_county.merge(right=addressData, how='left', on='Parcel_ID')
    fp.write('Fairfax_County_VA_clean.parq', fairfax_county)
    fairfax_county.to_csv("Fairfax_County_VA_clean.csv", index=False)
    