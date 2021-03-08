from abc import ABC, abstractmethod
import sklearn as sk
import pandas as pd
import numpy as np

############## Funcion auxiliar ##################
# Devuelve una lista con las columnas no binarias
def columns_no_binary(df):
    columns = []
    for c in df.columns:
        values = df[c].value_counts().size
        if values > 2:
            columns.append(c)
    
    return columns

#################### Clases ######################
# Clase abstracta
class Sub_preprocessing(ABC):
    @abstractmethod
    def fit(self, df):
        pass
    
    def transform(self, df):
        pass
    
    def function(self):
        pass

# Elimina las columnas con alta cardinalidad del df
class Drop_high_cardinals(Sub_preprocessing):
    def fit(self, df):
        #Do nothing
        return
    
    def transform(self, df):
        df = df.drop(axis = 1, columns = [
            'nombre', 'id_usuario', 'id_ticket'
        ])
        return df
    
    def function(self):
        print("Elimina las columnas con alta cardinalidad")

# Transforma las columnas categoricas en dummy variables
class Dummy_variables(Sub_preprocessing):
    def fit(self, df):
        cols_with_nan = []
        cols_without_nan = []
        for c in df.columns:
            # si la columna tiene null
            null = df[c].isnull().any()
            # si tiene menos de 4 valores
            categorical = (df[c].value_counts().size < 4)
            
            if categorical:
                if null:
                    cols_with_nan.append(c)
                else:
                    cols_without_nan.append(c)
                    
        self.cols_with_nan = cols_with_nan
        self.cols_without_nan = cols_without_nan
        return
    
    def transform(self, df):
        if 'fila' in df.columns:
            # Para resolver el caso de valor extra en el holdout
            df['fila'] = df['fila'].replace('atras', np.nan)
            
        df = pd.get_dummies(df, drop_first=True, columns=self.cols_without_nan)
        df = pd.get_dummies(df, drop_first=True, dummy_na=True, columns=self.cols_with_nan)
        return df
    
    def function(self):
        print("Transforma las variables categoricas en dummy variables")

# Elimina la columna con muchos nan (mayor igual al 70%)
class Drop_column_nan(Sub_preprocessing):
    def fit(self, df):
        columns = []
        for column in df.columns:
            nulls = df[column].isnull().sum() * 100 / df[column].index.size
            if nulls >= 70 :
                columns.append(column)
                
        self.columns = columns
        return
    
    def transform(self, df):
                
        return df.drop(axis = 1 ,columns = self.columns)
    
    def function(self):
        print("Elimina las columnas con un porcentaje de valores nan mayor igual al 70%")

# Rellena los nan de edad con 0
class Nan_to_zero(Sub_preprocessing):
    def fit(self, df):
        #Do nothing
        return
    
    def transform(self, df):
        df['edad'] = df['edad'].replace(np.nan, 0)
        return df
    
    def function(self):
        print("Rellena los nan de edad con ceros")

# Rellena los nan de edad con la media
class Nan_to_mean(Sub_preprocessing):
    def fit(self, df):
        edad = df['edad']
        self.mean = edad.mean()
    
    def transform(self, df):
        df['edad'] = df['edad'].replace(np.nan, self.mean)
        return df
    
    def function(self):
        print("Rellena los nan de edad con la media")

# Rellena los nan de edad con la moda
class Nan_to_mode(Sub_preprocessing):
    def fit(self, df):
        edad = df['edad']
        self.mode = edad.mode()[0]
    
    def transform(self, df):
        df['edad'] = df['edad'].replace(np.nan, self.mode)
        return df
    
    def function(self):
        print("Rellena los nan de edad con la moda")

# Rellena los nan de edad con la mediana
class Nan_to_median(Sub_preprocessing):
    def fit(self, df):
        edad = df['edad']
        self.median = edad.median()
    
    def transform(self, df):
        df['edad'] = df['edad'].replace(np.nan, self.median)
        return df
    
    def function(self):
        print("Rellena los nan de edad con la mediana")

# Normaliza los atributos no binarios
class Normalizer_columns(Sub_preprocessing):
    def fit(self, df):
        self.columns = columns_no_binary(df)
        return
    
    def transform(self, df):
        columns = self.columns
        df_columns = df[columns]
        normalizer = sk.preprocessing.Normalizer()
        df[columns] = normalizer.fit_transform(df_columns) 
        return df
    
    def function(self):
        print("Normaliza los atributos no binarios")
        
# Estandariza los atributos no binarios
class Std_columns(Sub_preprocessing):
    def fit(self, df):
        self.columns = columns_no_binary(df)
        df_columns = df[self.columns]
        self.min_max_scaler = sk.preprocessing.MinMaxScaler()
        self.min_max_scaler.fit(df_columns) 
        
    def transform(self, df):
        columns = self.columns
        df_columns = df[columns]
        df[columns] = self.min_max_scaler.transform(df_columns) 
        return df
    
    def function(self):
        print("Estandariza los atributos no binarios")

# Mezcla los diferentes preprocesados
class Preprocessing(Sub_preprocessing):

    def __init__(self, list_sub_preprocessing):
        self.list_sub_preprocessing = list_sub_preprocessing

    def fit(self, df):
        for p in self.list_sub_preprocessing:
            p.fit(df)
            df = p.transform(df)
        
    def transform(self, df):
        for p in self.list_sub_preprocessing:
            df = p.transform(df)
        
        return df
    
    def function(self):
        for p in self.list_sub_preprocessing:
            p.function()
    
######################### Mejores preprocesados ###########################

# Elimina las columnas con alta cardinalidad
# Elimina las columnas con un porcentaje de valores nan mayor igual al 70%
# Transforma las variables categoricas en dummy variables
# Rellena los nan de edad con la media
class Preprocessing_Tree_Bagging(Preprocessing):
    
    def __init__(self):
        Preprocessing.__init__(self, [
            Drop_high_cardinals(), 
            Drop_column_nan(),
            Dummy_variables(), 
            Nan_to_mean()
        ])
    
    def fit(self, df):
        Preprocessing.fit(self, df)
    
    def transform(self, df):
        return Preprocessing.transform(self, df)
    
    def function(self):
        Preprocessing.function(self)

# Elimina las columnas con alta cardinalidad
# Elimina las columnas con un porcentaje de valores nan mayor igual al 70%
# Transforma las variables categoricas en dummy variables
# Rellena los nan de edad con la moda
# Estandariza los atributos no binarios
class Preprocessing_KNN_SVM(Preprocessing):
    
    def __init__(self):
        Preprocessing.__init__(self, [
            Drop_high_cardinals(), 
            Drop_column_nan(),
            Dummy_variables(), 
            Nan_to_mode(),
            Std_columns()
        ])
    
    def fit(self, df):
        Preprocessing.fit(self, df)
    
    def transform(self, df):
        return Preprocessing.transform(self, df)
    
    def function(self):
        Preprocessing.function(self)

# Elimina las columnas con alta cardinalidad
# Elimina las columnas con un porcentaje de valores nan mayor al 70%
# Transforma las variables categoricas en dummy variables
# Rellena los nan de edad con la moda
class Preprocessing_NB_Boosting_Stacking(Preprocessing):
    
    def __init__(self):
        Preprocessing.__init__(self, [
            Drop_high_cardinals(), 
            Drop_column_nan(),
            Dummy_variables(), 
            Nan_to_mode()
        ])
        
    def fit(self, df):
        Preprocessing.fit(self, df)
    
    def transform(self, df):
        return Preprocessing.transform(self, df)
    
    def function(self):
        Preprocessing.function(self)
        
# Elimina las columnas con alta cardinalidad
# Elimina las columnas con un porcentaje de valores nan mayor igual al 70%
# Transforma las variables categoricas en dummy variables
# Rellena los nan de edad con ceros
class Preprocessing_RF(Preprocessing):
    
    def __init__(self):
        Preprocessing.__init__(self, [
            Drop_high_cardinals(), 
            Drop_column_nan(),
            Dummy_variables(), 
            Nan_to_zero()
        ])
    
    def fit(self, df):
        Preprocessing.fit(self, df)
    
    def transform(self, df):
        return Preprocessing.transform(self, df)
    
    def function(self):
        Preprocessing.function(self)
        