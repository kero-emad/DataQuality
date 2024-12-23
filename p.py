import io
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd
import numpy as np
from langchain_ollama import ChatOllama



def calculate_outliers(df):
    outlier_info = {}
    for col in df.select_dtypes(include=['number']):
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        if not outliers.empty:
            outlier_info[col] = len(outliers)
    return outlier_info


def extract_column_name(question, columns):
    for col in columns:
        if col.lower() in question.lower():
            return col
    return None


def handle_dataset_question(question, dataset_context):
    columns = dataset_context["columns"]
    summary = dataset_context["summary"]
    missing = dataset_context["missing"]
    unique_values = dataset_context["unique_values"]

    if "columns" in question.lower():
        return f"The dataset has the following columns: {', '.join(columns)}."
    elif "missing" in question.lower():
        missing_info = "\n".join([f"{col}: {count} missing values" for col, count in missing.items() if count > 0])
        return missing_info if missing_info else "There are no missing values in the dataset."
    elif "unique values" in question.lower():
        col_name = extract_column_name(question, columns)
        if col_name:
            return f"The column '{col_name}' has the following unique values: {unique_values[col_name]}"
        else:
            return "Please specify a valid column to get unique values."
    elif "duplicates" in question.lower():
        duplicate_count = df.duplicated().sum()
        return f"The dataset has {duplicate_count} duplicate rows." if duplicate_count > 0 else "There are no duplicate rows in the dataset."
    elif "rows" in question.lower():
        return f"The dataset contains {len(df)} rows."
    elif "summary" in question.lower():
        return "Summary statistics:\n" + "\n".join([f"{col}: {stats}" for col, stats in summary.items()])
    elif "outliers" in question.lower():
        outlier_info = calculate_outliers(df)
        if outlier_info:
            return "Outliers detected:\n" + "\n".join([f"{col}: {count} outliers" for col, count in outlier_info.items()])
        else:
            return "No outliers detected in the numerical columns."
            
    elif "info" in question.lower():
        num_rows = len(df)
        num_columns = len(columns)
        num_categorical = len(df.select_dtypes(include=['object', 'category']).columns)
        num_numerical = len(df.select_dtypes(include=['number']).columns)

        return (
            f"The dataset contains:\n"
            f"- {num_rows} rows\n"
            f"- {num_columns} columns\n"
            f"  - {num_categorical} categorical columns\n"
            f"  - {num_numerical} numerical columns"
        )
    elif "describe" in question.lower():
        num_rows = len(df)
        num_columns = len(columns)
        num_categorical = len(df.select_dtypes(include=['object', 'category']).columns)
        num_numerical = len(df.select_dtypes(include=['number']).columns)

        return (
            f"The dataset contains:\n"
            f"- {num_rows} rows\n"
            f"- {num_columns} columns\n"
            f"  - {num_categorical} categorical columns\n"
            f"  - {num_numerical} numerical columns"
        )

    else:
        return "I'm not sure about that. Please ask specific questions about the dataset."

st.set_page_config(layout="wide") 
st.title("Data Quality Project")

if "log" not in st.session_state:
    st.session_state.log = []

uploaded_file = st.sidebar.file_uploader("Upload your file", type=["csv"])

if uploaded_file is not None:
    if 'df' not in st.session_state:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.session_state.log.append("Dataset uploaded")
    df = st.session_state.df

    st.sidebar.subheader("Data Operations")
    operation = st.sidebar.selectbox(
        "Choose an operation:",
        [
            "View Dataset Info",
            "Describe Dataset",
            "View Dataset",
            "Handle Invalid Numeric Data",
            "Handle Missing Values",
            "Handle Duplicate Rows",
            "Handle Column Names",
            "Data Type Convert",
            "Drop Columns",
            "Handle Classification Columns",
            "Visualization",
            "Handle outlayers",
            "Correlation",
            "Interactions",
            "View log",
            "Before Download",
            "Chat",
            "ollama"
        ],
    )

    if operation == "View Dataset Info":
        info_data = {
            "Column": df.columns,
            "Not null count": df.count().values,
            "Data type": [df[col].dtype for col in df.columns],
        }
        info_df = pd.DataFrame(info_data)
        st.subheader("Dataset Information")
        st.dataframe(info_df)
        st.session_state.log.append("Viewed dataset info")

    elif operation == "Describe Dataset":
        st.subheader("Summary")
        st.write(df.describe())
        st.session_state.log.append("Viewed dataset description")

    elif operation == "View Dataset":
        if st.button("Head"):
            st.write(df.head())
            st.session_state.log.append("Viewed the first 5 rows in the dataset")
        elif st.button("Tail"):
            st.write(df.tail())
            st.session_state.log.append("Viewed the last 5 rows in the dataset")
        else:
            number=st.slider("select number of rows to display",1,df.shape[0])
            st.write(df.head(number))
            st.session_state.log.append(f"Viewed the first {number} rows of the dataset.")
        

    elif operation == "Handle Invalid Numeric Data":
        numeric_cols = st.multiselect(
            "Select columns that should be numeric:",
            options=df.columns.to_list()
        )

        if numeric_cols:
            invalid_cols = {}
            for col in numeric_cols:
                invalid_values = df[col].apply(
                    lambda x: isinstance(x, str) and not str(x).replace('.', '', 1).isdigit()
                )
                if invalid_values.any():
                    invalid_cols[col] = df[invalid_values].index.tolist()

            if invalid_cols:
                st.subheader("Invalid Data Found")
                for col, indices in invalid_cols.items():
                    st.write(f"**Column:** {col}")
                    st.write(f"**Invalid Values:** {df.loc[indices, col].unique()}")
                if st.button("Replace Invalid Values with NaN"):
                    for col in invalid_cols:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    st.session_state.df = df
                    st.success("Invalid values replaced with NaN.")
                    st.session_state.log.append(f"Handled invalid data in {col} column ")
            else:
                st.success("All selected columns have valid numeric data.")

    elif operation == "Handle Missing Values":
        missing_values = df.isnull().sum()
        st.write("Missing Values per Column:")
        st.table(missing_values)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.isnull(), cmap="viridis", cbar=True, ax=ax)
        plt.title("Missing Values Heatmap")
        st.pyplot(fig)
        empty_cols = [col for col in df.columns if df[col].isnull().any()]
        if empty_cols:
            selected_col = st.selectbox("Select a column with missing values:", empty_cols)
            col_type = df[selected_col].dtype

       
            if np.issubdtype(col_type, np.number):
                 fill_method = st.selectbox(
                "Choose a method to handle missing values:",
                ["Mean", "Median", "Mode", "Delete"]
                )
            elif np.issubdtype(col_type, object):
                fill_method = st.selectbox(
                "Choose a method to handle missing values:",
                ["Mode", "Delete"]
               )

            if fill_method != "Delete":
                 if fill_method == "Mean":
                    fill_value = df[selected_col].mean()
                 elif fill_method == "Median":
                    fill_value = df[selected_col].median()
                 elif fill_method == "Mode":
                    fill_value = df[selected_col].mode()[0]
            
                 
                 st.write(f"The value to be used for imputation in {selected_col} is: **{fill_value}**")

    
            num_missing = df[selected_col].isnull().sum()
            st.write(f"Number of missing rows in {selected_col}: **{num_missing}**")
            st.write(df[df[selected_col].isnull()])

            if st.button("Apply Changes"):
              if fill_method == "Delete":
                 df = df.dropna(subset=[selected_col])  
                 st.success(f"Rows with missing values in {selected_col} have been deleted")
                 missing_values = df.isnull().sum()
                 st.table(missing_values)
                 fig, ax = plt.subplots(figsize=(10, 6))
                 sns.heatmap(df.isnull(), cmap="viridis", cbar=True, ax=ax)
                 plt.title("Missing Values Heatmap")
                 st.pyplot(fig)
                 st.session_state.log.append(f"Handled missing values by dropping rows with missing values in column {selected_col}")
              else:
                df[selected_col].fillna(fill_value, inplace=True)
                st.success(f"Missing values in {selected_col} have been replaced with {fill_value}")
                missing_values = df.isnull().sum()
                st.table(missing_values)
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(df.isnull(), cmap="viridis", cbar=True, ax=ax)
                plt.title("Missing Values Heatmap")
                st.pyplot(fig)
                st.session_state.log.append(f"Handled missing values by replacing missing values in column {selected_col} with {fill_method} value")


            
              st.session_state.df = df
        else:
          st.info("No columns with missing values")


            
    elif operation == "Handle Duplicate Rows":
        duplicated_rows = df[df.duplicated()]
        duplicate_count = len(duplicated_rows)

        if duplicate_count > 0:
            st.subheader("Duplicate Rows Found")
            st.write(f"Total Duplicate Rows: {duplicate_count}")
            st.write(duplicated_rows)

            if st.button("Delete Duplicate Rows"):
                df.drop_duplicates(inplace=True)
                st.session_state.df = df
                st.success("Duplicate rows have been deleted")
                duplicated_rows = df[df.duplicated()]
                duplicate_count = len(duplicated_rows)
                st.write(f"Total Duplicate Rows: {duplicate_count}")
                st.write(duplicated_rows)
                st.session_state.log.append("Duplicates rows have been deleted")

        else:
            st.info("No duplicate rows found.")

    elif operation == "Handle Column Names":
        st.subheader("Change Column Names")
        column_names = df.columns.tolist()
        selected_col = st.selectbox("Select a column to rename:", column_names)

        new_name = st.text_input(f"Enter new name for '{selected_col}':", selected_col)

        if st.button("Change Name"):
            df.rename(columns={selected_col: new_name},inplace=True)
            st.session_state.df = df
            st.success(f"Column {selected_col} has been renamed to {new_name}")
            st.write(df.columns)
            st.session_state.log.append(f"Column {selected_col} has been renamed to {new_name}")


    elif operation == "Drop Columns":
        st.subheader("Drop Columns from Dataset")
        columns = df.columns.tolist()
        if columns:
           selected_columns = st.multiselect("Select column to drop:", columns)
         
           if selected_columns:
             st.write(f"Are You sure that you wamt to drop {selected_columns}")
             if st.button("Apply Drop"):
                df.drop(columns=selected_columns, inplace=True)
                st.session_state.df = df
                st.success(f"Successfully dropped column:{selected_columns}")
                st.write(df.columns)
                st.session_state.log.append(f"Column {selected_columns} has been dropped")
           else:
             st.info("select column to drop")
        else:
             st.info("No columns")


    elif operation == "Data Type Convert":
       st.header("Convert Data Types")
       st.write("Use this tool to change the data type of a column.")
    
    
       selected_column = st.selectbox("Select a column", df.columns)
       current_dtype = df[selected_column].dtype
    
    
       st.write(f"Current data type of {selected_column}  {current_dtype}")
    
    
       convert_options = ["int", "float", "string", "datetime"]
       target_dtype = st.selectbox("Select the target data type:", convert_options)
    
   
       st.subheader("Data Before Conversion")
       st.dataframe(df[[selected_column]].head(10))
    
   
       if st.button("Apply Conversion"):
          try:
             if target_dtype == "int":
                df[selected_column] = pd.to_numeric(df[selected_column], errors='coerce').fillna(0).astype(int)
             elif target_dtype == "float":
                df[selected_column] = pd.to_numeric(df[selected_column], errors='coerce').fillna(0.0)
             elif target_dtype == "string":
                df[selected_column] = df[selected_column].astype(str)
             elif target_dtype == "datetime":
                df[selected_column] = pd.to_datetime(df[selected_column], errors='coerce')
            
            
             st.subheader("Data After Conversion")
             st.dataframe(df[[selected_column]].head(10))
             st.session_state.df = df
             st.success(f"Successfully converted '{selected_column}' to {target_dtype}.")
             st.session_state.log.append(f"Column {selected_column} has been converted to {target_dtype}")

          except Exception as e:
             st.error(f"Failed to convert '{selected_column}' to {target_dtype}. Error: {str(e)}")
 


    elif operation == "Handle Classification Columns":
         st.subheader("Convert categorical Columns to Numerical")
         categorical_cols = df.select_dtypes(include=['object']).columns
         if not categorical_cols.any():
             st.warning("No classification columns found in the dataset")
         else:
             selected_col = st.selectbox("Select a classification column:", categorical_cols)
             unique_values = df[selected_col].unique()
             mapping = {value: idx for idx, value in enumerate(unique_values)}
             st.write("Mapping")
             st.write(mapping)

             if st.button("Apply Mapping"):
               df[selected_col] = df[selected_col].map(mapping)
               st.session_state.df = df
               st.success(f"Column {selected_col} is converted to numerical")
               st.session_state.log.append(f"Column {selected_col} has been converted to numerical")



    elif operation == "Visualization":
       selected_col = st.selectbox("Select a column for visualization:", df.columns)
       col_type = df[selected_col].dtype

       if np.issubdtype(col_type, np.number):
           st.subheader(f"Histogram for {selected_col}")
           fig, ax = plt.subplots(figsize=(30, 6))
           ax.hist(df[selected_col].dropna(), bins=20, color='skyblue', edgecolor='black')
           ax.set_xlabel(selected_col)
           ax.set_ylabel("Frequency")
           ax.set_title(f"Histogram of {selected_col}")
           st.pyplot(fig)
           st.session_state.log.append(f"View histogram for column {selected_col}")
           

       elif np.issubdtype(col_type, object): 
           st.subheader(f"Pie Chart for {selected_col}")
           pie_data = df[selected_col].value_counts() 
           fig, ax = plt.subplots(figsize=(70, 6))
           ax.pie(
            pie_data,
            labels=pie_data.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=sns.color_palette('pastel')[0:len(pie_data)]
        )
           ax.set_title(f"Pie Chart of {selected_col}")
           st.pyplot(fig)
           st.session_state.log.append(f"View piechart for column {selected_col}")

       else:
        st.warning(f"Visualization is not supported for columns of type: {col_type}")  




    elif operation == "Handle outlayers":
       numeric_cols = [col for col in df.columns if np.issubdtype(df[col].dtype, np.number)]
       if numeric_cols:
           selected_col = st.selectbox("Select a numeric column for outlier handling:", numeric_cols)
           st.subheader(f"Box Plot for {selected_col}")
           fig, ax = plt.subplots()
           sns.boxplot(x=df[selected_col], ax=ax, color='skyblue')
           ax.set_title(f"Box Plot of {selected_col} (Before)")
           st.pyplot(fig)
           Q1 = df[selected_col].quantile(0.25)
           Q3 = df[selected_col].quantile(0.75)
           IQR = Q3 - Q1
           lower_bound = Q1 - 1.5 * IQR
           upper_bound = Q3 + 1.5 * IQR
           outliers = df[(df[selected_col] < lower_bound) | (df[selected_col] > upper_bound)]
           if len(outliers) > 0:
               st.write(f"**Number of Outliers:** {len(outliers)}")
               st.dataframe(outliers)

               action = st.radio("Choose an action:", ["Drop Outliers", "Clip Outliers"])

               if action == "Clip Outliers":
                  clip_lower = max(lower_bound, df[selected_col].min())
                  clip_upper = min(upper_bound, df[selected_col].max())
                  st.write(f"**Values will be clipped to the range:** [{clip_lower}, {clip_upper}]")

        
               if st.button("Apply Changes"):
                   if action == "Drop Outliers":
                      df = df[~((df[selected_col] < lower_bound) | (df[selected_col] > upper_bound))]
                      st.success("Outliers dropped successfully")
                      st.session_state.log.append("outliers have been deleted")
                   elif action == "Clip Outliers":
                      df[selected_col] = df[selected_col].clip(lower=lower_bound, upper=upper_bound)
                      st.success("Outliers clipped successfully")
                      st.session_state.log.append("outliers have been clipped")

                   st.session_state.df = df
                   st.subheader(f"Box Plot for {selected_col} (After)")
                   fig, ax = plt.subplots()
                   sns.boxplot(x=df[selected_col], ax=ax, color='skyblue')
                   ax.set_title(f"Box Plot of {selected_col} (After)")
                   st.pyplot(fig)
           else:
               st.success(f"No outliers found in the column '{selected_col}'.")

       else:
          st.warning("No numeric columns available for outlier handling.")


    elif operation == "Interactions":
       st.subheader("Interactions")
       numerical_cols = df.select_dtypes(include=['number']).columns

       if len(numerical_cols) >= 2:
            col1 = st.selectbox("Select the first numerical column:", numerical_cols)
            col2 = st.selectbox("Select the second numerical column:", numerical_cols)
            if col1 != col2:
               st.subheader(f"Interaction Between {col1} and {col2}")
               fig, ax = plt.subplots(figsize=(10, 6))
               ax.scatter(df[col1], df[col2], alpha=0.7, color='skyblue', edgecolor='black')
               ax.set_xlabel(col1)
               ax.set_ylabel(col2)
               ax.set_title(f"{col1} vs {col2}")
               st.pyplot(fig)
               st.session_state.log.append(f"view interaction between {col1},{col2}")
            else:
               st.warning("Please select two different columns.")
       else:
               st.warning("Not numerical columns to make interactions")
       
       

    elif operation == "Correlation":
       numeric_cols = df.select_dtypes(include=[np.number])

       if numeric_cols.shape[1] > 1: 
          st.subheader("Correlation Heatmap")
          correlation_matrix = numeric_cols.corr()
          fig, ax = plt.subplots(figsize=(10, 8))
          sns.heatmap(
            correlation_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            cbar=True,
            square=True,
            ax=ax
        )
          ax.set_title("Correlation Heatmap")
          st.pyplot(fig)
          st.session_state.log.append("View Heat map for numerical columns")
       else:
        st.warning("No numeric columns to calculate correlation")

    elif operation == "View log":
        st.subheader("User Actions Log")
        if st.session_state.log:
            for idx, action in enumerate(st.session_state.log, start=1):
                st.write(f"{idx}. {action}")
        else:
            st.info("No actions performed yet")

    elif operation == "Before Download":
       st.subheader("Dataset Overview")
       num_variables = len(df.columns)
       num_rows = len(df)
       missing_cells = df.isnull().sum().sum()
       missing_cells_pct = (missing_cells / (num_variables * num_rows)) * 100 if num_rows > 0 else 0

       duplicate_rows = df.duplicated().sum()
       duplicate_rows_pct = (duplicate_rows / num_rows) * 100 if num_rows > 0 else 0

    
       total_size = df.memory_usage(deep=True).sum()
       avg_row_size = total_size / num_rows if num_rows > 0 else 0

       num_numerical = len(df.select_dtypes(include=['number']).columns)
       num_categorical = len(df.select_dtypes(include=['object', 'category']).columns)

       st.markdown("### Overview")
       st.markdown(f"- **Number of Variables:** {num_variables}")
       st.markdown(f"- **Number of Rows:** {num_rows}")
       st.markdown(f"- **Missing Cells:** {missing_cells}")
       st.markdown(f"- **Missing Cells (%):** {missing_cells_pct:.1f}%")
       st.markdown(f"- **Duplicate Rows:** {duplicate_rows}")
       st.markdown(f"- **Duplicate Rows (%):** {duplicate_rows_pct:.1f}%")
       st.markdown(f"- **Total Size in Memory:** {total_size / 1024:.1f} KB")
       st.markdown(f"- **Average Row Size in Memory:** {avg_row_size:.1f} B")
       st.markdown("### Variable Types")
       st.markdown(f"- **Numerical:** {num_numerical}")
       st.markdown(f"- **Categorical:** {num_categorical}")

       st.title('Visit us on github "https://github.com/kero-emad/DataQuality" ')

       new_data = df.to_csv(index=False)
       st.download_button(
       label="Download Dataset",
       data=new_data,
       file_name="cleaned_dataset.csv",
       mime="text/csv"
        )      

    elif operation == "Chat":
      if "messages" not in st.session_state:
            st.session_state["messages"] = [
            {"role": "assistant", "content": "Hi! I can help you with questions about your dataset. Please ask about columns, values, or any dataset-related insights."}
            ]

      for msg in st.session_state["messages"]:
            st.chat_message(msg["role"]).write(msg["content"])

      if 'df' in st.session_state:
        df = st.session_state["df"]
        dataset_context = {
            "columns": df.columns.tolist(),
            "summary": df.describe(include='all').to_dict(),
            "missing": df.isnull().sum().to_dict(),
            "unique_values": {col: df[col].unique().tolist() for col in df.columns},
            "dataframe": df

        }
      else:
           dataset_context = None

    
      if prompt := st.chat_input():
          st.session_state["messages"].append({"role": "user", "content": prompt})
          st.chat_message("user").write(prompt)

        
          if dataset_context:
            response_content = handle_dataset_question(prompt, dataset_context)
          else:
            response_content = "No dataset is currently loaded. Please upload a dataset to ask specific questions."

          st.chat_message("assistant").write(response_content)
          st.session_state["messages"].append({"role": "assistant", "content": response_content})


    elif operation == "ollama":
     if "messages" not in st.session_state:
         st.session_state["messages"] = [{"role": "assistant", "content": "Hi there! How can I assist you with your dataset today?"}]

     for msg in st.session_state["messages"]:
         st.chat_message(msg["role"]).write(msg["content"])

     if 'df' not in st.session_state:
         st.info("Please upload a dataset to enable Chat functionality.")
     else:
        df = st.session_state["df"]
        if prompt := st.chat_input():
             st.session_state["messages"].append({"role": "user", "content": prompt})
             st.chat_message("user").write(prompt)

             dataset_summary = f"The dataset contains {df.shape[0]} rows and {df.shape[1]} columns. Columns: {', '.join(df.columns)}."
             if df.isnull().values.any():
                missing_summary = f" It has missing values in columns: {', '.join(df.columns[df.isnull().any()].tolist())}."
             else:
                missing_summary = " There are no missing values in the dataset."
             dataset_info = dataset_summary + missing_summary

             assistant_message = (
                f"Here is some information about the dataset: {dataset_info} "
                f"Feel free to ask any questions related to the dataset."
            )
             st.session_state["messages"].append({"role": "assistant", "content": assistant_message})
             st.chat_message("assistant").write(assistant_message)

             llm = ChatOllama(
                model="llama3.2",
                temperature=0,
             )
             try:
                response = llm.invoke(st.session_state["messages"])
                st.chat_message("assistant").write(response.content)
                st.session_state["messages"].append({"role": "assistant", "content": response.content})
             except Exception as e:
                st.error(f"An error occurred: {e}")

else:
    st.sidebar.error("Please upload dataset")




