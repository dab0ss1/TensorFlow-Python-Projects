# Dependencies
import pandas as pd
from sklearn import svm

# Packages for visuals
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(font_scale=1.2)

# ignore deprecation warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Load in fruit data (normalized to 100 garms)
content = pd.read_csv('fruit_data.csv')
print(content)

# Plot two ingredients
sns.lmplot('Calories', 'Potassium', data=content, hue='Type',
           palette='Set1', fit_reg=False, scatter_kws={"s": 70})
plt.show()

# Model inputs
ingredients = content[['Calories', 'Fat', 'Cholesterol', 'Sodium', 'Potassium', 'Fiber', 'Sugar', 'Protien']].as_matrix()
unique_type_label = content['Type'].unique().tolist()
type_label = []
for x in range(len(content['Type'])):
    type_label.append(unique_type_label.index(content['Type'][x]))
print(type_label)

# Feature names
fruit_features = content.columns.values[1:].tolist()
print(fruit_features)

# Fit the SVM model
model = svm.SVC(kernel='linear', decision_function_shape='ovo')
model.fit(ingredients, type_label)

# Create a function to guess the fruit
def find_fruit(parts):
    print('You\'re looking at a ', unique_type_label[int(model.predict([parts]))])

# Predict Apple
find_fruit([55.0,0.25,0.0,0.0015,0.115,2.5,10.4,0.2])

# Predict Banana
find_fruit([80.0,0.3,0.0,0.01,0.258,2.0,10.0,1.1])

# Predict Orange
find_fruit([49.0,0.15,0.0,0.0,0.200,2.276,8.0,0.83])