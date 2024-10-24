#Write a script to perform patient matching
import pandas as pd
import altair as alt
#We will want this script to take a dataframe of patients with their IDs, Diagnosed Gene And a list of associated proteins
#The scrpit will also take a similarity matrix 

#Function for primary level matching only
def patient_matching (diagnosis_file:pd.DataFrame, similarity_table:pd.DataFrame) -> None:
    if not isinstance (similarity_table, pd.DataFrame) :
        raise TypeError("Expected a pandas DataFrame")
    
    #Since some patients have the same diagnosis, we will create two dictionaries
    #One with patient IDs and gene diagnosis and the other with patinet ID and associated genes
    #Keep in mind that we will have to loop in the similarity table by patient ID not number since that has been changed by the deletion of some patients 
    #with multiple diagnosed genes

    #I think it will be a good idea to add an index to the similarity table 
    diagnosis_dict = diagnosis_file[['ID','Gene']].to_dict(orient='records')
    diagnosis_dict = {item['ID'] : item['Gene'] for item in diagnosis_dict}


    #Make a patient ID and associated gene disctionary
    associated_genes_dict = diagnosis_file[['ID','Associated Proteins']].to_dict(orient='records')
    associated_genes_dict = {item['ID'] : item['Associated Proteins'] for item in associated_genes_dict}

    #Define a list of patients to loop through
    patient_list = diagnosis_file['ID'].tolist()
    #patient_list = patient_list[ : 50] #shorten list for now
    j = 0
    #drop patients with multiple diagnosis in the similarity table
    #675 patients with a single gene diagnosis
    matching_data = [] # We will do a list of tuples instead otherwise the dataframe will take ages to populate
    k = 0 #Use k as the index in matching data

  
    for i in similarity_table.index:
        if i not in patient_list:
            #Drop row and coloumn of every patient id that is not in the patient list
            similarity_table = similarity_table.drop(columns=i).drop([i])


    match_table = similarity_table.copy()
    match_table = match_table.astype(str)

    for i in patient_list:
        patient_diagnosis = diagnosis_dict[i] #get gene diagnosis
        patient_interacting_genes = associated_genes_dict[i] #get associated genes
        sim_table_subset = similarity_table[i][:] #filter out the list of patients from the similairty matrix
        sim_table_subset = sim_table_subset.drop([i]) #drop the patinet themsevels since their value will be 1
        sim_table_subset = sim_table_subset.sort_values(ascending=False) #put values in decending order such that 1 is higher match and 674 is lowest

        patient2_list = sim_table_subset.index.tolist() #Get the IDs of other patients in decending order 
        #patient2_scores = sim_table_subset.values.tolist() #Get the scores of other patinets in decending order, we will just vaalues from sim_table_subset
        rank = 1
        for j in patient2_list:

            patient2_id = j
            patient2_interacting_genes = associated_genes_dict[j]
            patient2_diagnosis = diagnosis_dict[j]
            patient2_score = sim_table_subset[j]

        
            #We will only consider same gene and primary interactor matches for this work    
            if patient2_diagnosis == patient_diagnosis:
                match_table.loc[i , j] = 'same_gene'
                gene_class = 'same_gene'
            elif patient2_diagnosis in patient_interacting_genes:
                match_table.loc[i , j] = 'interacting_gene'
                gene_class = 'interacting_gene'
            else:
                match_table.loc[i , j] = 'no_match'
                gene_class = 'no_match'
            matching_data.append((i , j , patient2_score, gene_class, rank))
            k = k + 1
            rank = rank + 1

    #Convert matching data to a dataframe
    matching_data = pd.DataFrame(matching_data, columns=['Patient 1', 'Patient 2', 'Score', 'Match Class', 'Rank'])

    #Drop the permutations of patient matches. For example UDN234 - UDN344 is the same as UDN344 and UDN234
    matching_data['sorted'] = matching_data.apply(lambda row: tuple(sorted([row['Patient 1'], row['Patient 2']])), axis = 1)
    # Drop duplicates based on the sorted column
    sorted_match_data = matching_data.drop_duplicates(subset='sorted')
    # Drop the temporary 'sorted' column
    sorted_match_data = sorted_match_data.drop(columns='sorted')
    return matching_data, sorted_match_data, match_table

#Function for primary and secondary level matching
def patient_matching2 (diagnosis_file:pd.DataFrame, similarity_table:pd.DataFrame) -> None:
    if not isinstance (similarity_table, pd.DataFrame) :
        raise TypeError("Expected a pandas DataFrame")
    
    #Since some patients have the same diagnosis, we will create two dictionaries
    #One with patient IDs and gene diagnosis and the other with patinet ID and associated genes
    #Keep in mind that we will have to loop in the similarity table by patient ID not number since that has been changed by the deletion of some patients 
    #with multiple diagnosed genes

    #I think it will be a good idea to add an index to the similarity table 
    diagnosis_dict = diagnosis_file[['ID','Gene']].to_dict(orient='records')
    diagnosis_dict = {item['ID'] : item['Gene'] for item in diagnosis_dict}


    #Make a patient ID and associated gene disctionary
    associated_genes_dict = diagnosis_file[['ID','Associated Proteins']].to_dict(orient='records')
    associated_genes_dict = {item['ID'] : item['Associated Proteins'] for item in associated_genes_dict}

    #Define a list of patients to loop through
    patient_list = diagnosis_file['ID'].tolist()
    #patient_list = patient_list[ : 50] #shorten list for now
    j = 0
    #drop patients with multiple diagnosis in the similarity table
    #675 patients with a single gene diagnosis
    matching_data = [] # We will do a list of tuples instead otherwise the dataframe will take ages to populate
    k = 0 #Use k as the index in matching data

  
    for i in similarity_table.index:
        if i not in patient_list:
            #Drop row and coloumn of every patient id that is not in the patient list
            similarity_table = similarity_table.drop(columns=i).drop([i])


    match_table = similarity_table.copy()
    match_table = match_table.astype(str)

    for i in patient_list:
        patient_diagnosis = diagnosis_dict[i] #get gene diagnosis
        patient_interacting_genes = associated_genes_dict[i] #get associated genes
        sim_table_subset = similarity_table[i][:] #filter out the list of patients from the similairty matrix
        sim_table_subset = sim_table_subset.drop([i]) #drop the patinet themsevels since their value will be 1
        sim_table_subset = sim_table_subset.sort_values(ascending=False) #put values in decending order such that 1 is higher match and 674 is lowest

        patient2_list = sim_table_subset.index.tolist() #Get the IDs of other patients in decending order 
        #patient2_scores = sim_table_subset.values.tolist() #Get the scores of other patinets in decending order, we will just vaalues from sim_table_subset
        rank = 1
        for j in patient2_list:

            patient2_id = j
            patient2_interacting_genes = associated_genes_dict[j]
            patient2_diagnosis = diagnosis_dict[j]
            patient2_score = sim_table_subset[j]

        
            #We will only consider same gene and primary interactor matches for this work    
            if patient2_diagnosis == patient_diagnosis:
                match_table.loc[i , j] = 'same_gene'
                gene_class = 'same_gene'
            elif patient2_diagnosis in patient_interacting_genes:
                match_table.loc[i , j] = 'interacting_gene'
                gene_class = 'interacting_gene'
            elif set(patient2_interacting_genes).intersection(set(patient_interacting_genes)):
                match_table.loc[i , j] = 'secondary_interaction'
                gene_class = 'secondary_interaction'
            else:
                match_table.loc[i , j] = 'no_match'
                gene_class = 'no_match'
            matching_data.append((i , j , patient2_score, gene_class, rank))
            k = k + 1
            rank = rank + 1

    #Convert matching data to a dataframe
    matching_data = pd.DataFrame(matching_data, columns=['Patient 1', 'Patient 2', 'Score', 'Match Class', 'Rank'])

    #Drop the permutations of patient matches. For example UDN234 - UDN344 is the same as UDN344 and UDN234
    matching_data['sorted'] = matching_data.apply(lambda row: tuple(sorted([row['Patient 1'], row['Patient 2']])), axis = 1)
    # Drop duplicates based on the sorted column
    sorted_match_data = matching_data.drop_duplicates(subset='sorted')
    # Drop the temporary 'sorted' column
    sorted_match_data = sorted_match_data.drop(columns='sorted')
    return matching_data, sorted_match_data

#Make a function for making a box plot 
def create_box_plot(sorted_match_data):
    alt.data_transformers.disable_max_rows()
    box_plot = alt.Chart(sorted_match_data[['Score','Match Class']]).mark_boxplot(size=50).encode(
        x=alt.X('Match Class:O', title='Match Class', sort=['no_match', 'same_gene', 'interacting_gene']),  # X-axis
        y=alt.Y('Score:Q', title='Phenotypic Score'),         # Y-axis
        color='Match Class:O'
        ).properties(
        width=300,
        height=300,
        title='Gene Interaction Box Plot'
        )

    # Calculate medians for each category
    median_values = sorted_match_data.groupby('Match Class').agg({'Score': 'median'}).reset_index()

    # Create a chart with median values
    median_text = alt.Chart(median_values).mark_text(
        align='center',
        baseline='middle',
        dy=-10,  # Adjust vertical position of text
        color='black'
    ).encode(
        x=alt.X('Match Class:O', title='Match Class', sort=['no_match', 'same_gene', 'interacting_gene']),
        y=alt.Y('Score:Q', title='Phenotypic Score'),
        text=alt.Text('Score:Q', format='.2g')  # Format the text to show median values
    )

    # Combine the box plot and median points
    final_plot = box_plot + median_text
    return final_plot

#Make a function for the rank and CDF power curve 

#Make a function for the Phenotypic score and CDF Power curve
def pheno_score_powercurve(matching_data):
    # Group by PatientID and MatchType, get the highest rank using idxmax
    idx = matching_data.groupby(['Patient 1', 'Match Class'])['Rank'].idxmin()

    # Filter the DataFrame to get only the highest rank for each patient and match type
    highest_rank_df = matching_data.loc[idx]

    dfs_by_match_type = {}
    for match_type in highest_rank_df['Match Class'].unique():
        # Filter by MatchType and store in the dictionary
        dfs_by_match_type[match_type] = highest_rank_df[highest_rank_df['Match Class'] == match_type]
    same_gene_ranks = dfs_by_match_type['same_gene'].sort_values(by='Score',ascending=False)
    interacting_gene_ranks = dfs_by_match_type['interacting_gene'].sort_values(by='Score',ascending=False)

    interacting_gene_patients = interacting_gene_ranks[~interacting_gene_ranks['Patient 1'].isin(same_gene_ranks['Patient 1'])]

    interacting_gene_patients['Match Class'] = 'interating_gene_only'

    #Concatenate Dataframes:
    combined_ranks = pd.concat([same_gene_ranks,interacting_gene_ranks, interacting_gene_patients])
    combined_ranks = combined_ranks.sort_values(['Match Class', 'Score'],ascending=False)

    # Step 4: Compute the cumulative proportion (CDF) for each group
    combined_ranks['CDF'] = 1 -  combined_ranks.groupby('Match Class')['Score'].rank(method='max', pct=True)


    # Step 3: Create the CDF plot using Altair
    cdf_plot = alt.Chart(combined_ranks).mark_line().encode(
        x=alt.X('Score:Q', scale=alt.Scale(reverse=True),axis=alt.Axis(title='Phenotypic Score')),    
        y='CDF:Q',
        color='Match Class:N'  # Use 'Group' to differentiate the lines
    ).properties(
        title='Cumulative Distribution Function (CDF) by Group',
        width=500,
        height=300
    )
    return cdf_plot

#Make a function for the dot plot curve too 
def create_dot_plot(diagnosis_file):
    chart = alt.Chart(diagnosis_file[['# Associated Proteins','ID']]).mark_point(filled=True).encode(
        y=alt.Y('# Associated Proteins:Q', title= '# of Associated Genes'),  # Use ordinal (discrete) scale for x-axis if values are categorical
        x = alt.X('ID:N', title='Patient ID'), # Count of occurrences on the y-axis
        tooltip=['# Associated Proteins', 'ID']
    ).properties(
        title='Dot Plot of Values'
    ).properties(
        title='# Of Associated Genes For Each Patient',
        width=600,  # Set the width of the plot
        height=400  # Set the height of the plot
    )
    return chart