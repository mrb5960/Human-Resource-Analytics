import pandas as pd
import matplotlib.pyplot as pplot
import numpy as np
import seaborn as sns

def main():
    path = "C:\\Study\\RIT\\BDA\\BDA_Term_Project\\HR_comma_sep.csv"
    df = pd.read_csv(path)
    # prints the first few rows of the dataset
    #print(df.head())
    # shows the statistics of the dataset
    print(df.describe())
    # shows data in a specific column
    #print(df.loc[:,['left', 'salary']])
    # slice of a dataset
    #print(df[0:30])

    # count of people who left in their salary range
    left_vs_salary = df.loc[:,['left','salary']].groupby('salary').sum()
    print('\n----------- left vs salary------------\n')
    print(left_vs_salary)

    # total number of people in each salary range
    total_in_each_salary_range = df.groupby('salary').size()
    print('\n------------Total in each salary range-------------\n')
    print(total_in_each_salary_range)

    # categories of salary
    salary_categories = df.salary.unique()
    print('\n------------Salary categories-----------\n')
    print(salary_categories)

    # list of values for total employees in each salary range
    total_in_each_salary_range_values = total_in_each_salary_range.values
    print('\n------------Total in each salary range values--------------\n')
    print(total_in_each_salary_range_values)

    # list of values for number of people left by salary range
    left_vs_salary_values = left_vs_salary['left'].values
    print('\n-------------List of left vs salary values-----------\n')
    print(left_vs_salary_values)

    # bar plot of salary range categories vs total number of employees in each category
    w = 0.3
    plot1 = pplot.bar(np.arange(0,len(salary_categories)) - w/2, total_in_each_salary_range, width=w, align='center')
    plot2 = pplot.bar(np.arange(0,len(salary_categories)) + w/2, left_vs_salary_values, width=w, align='center')
    pplot.xticks(np.arange(0, len(salary_categories)), salary_categories)
    pplot.legend((plot1[0], plot2[0]), ('Total', 'Left'), loc='upper left')
    pplot.title('Total and left employees')
    pplot.xlabel('Salary range')
    pplot.ylabel('Number of employees')
    pplot.show()

    # number of people left in a department
    left_vs_department = df.loc[:,['left', 'department']].groupby('department').sum()
    print('\n------------left vs department------------\n')
    print(left_vs_department)

    # total number of employees by department
    total_in_each_department = df.groupby('department').size()
    print('\n------------Total in each department------------\n')
    print(total_in_each_department)

    # categories in department column
    department_categories = df.department.unique()
    print('\n------------Department categories---------------\n')
    print(department_categories)

    # list of values for total employees in each department
    total_in_each_department_values = total_in_each_department.values
    print('\n------------Total in each department values--------------\n')
    print(total_in_each_department_values)

    # list of values for number of people left by department
    left_vs_department_values = left_vs_department['left'].values
    print('\n-------------List of left vs department values-----------\n')
    print(left_vs_department_values)

    # bar plot of department categories vs total number of employees in each category
    w = 0.3
    plot1 = pplot.bar(np.arange(0,len(department_categories)) - w/2, total_in_each_department, width=w, align='center')
    plot2 = pplot.bar(np.arange(0,len(department_categories)) + w/2, left_vs_department_values, width=w, align='center')
    pplot.xticks(np.arange(0, len(department_categories)), department_categories, rotation=45)
    pplot.legend((plot1[0], plot2[0]), ('Total', 'Left'), loc='upper left')
    pplot.title('Total and left employees')
    pplot.xlabel('Department')
    pplot.ylabel('Number of employees')
    pplot.tight_layout()
    pplot.show()

    # number of projects done by people left
    left_vs_projects = df.loc[:,['left','number_project']].groupby('number_project').sum()
    print('\n--------------left vs projects------------\n')
    print(left_vs_projects)

    # left_vs_project values
    left_vs_projects_values = left_vs_projects.values
    print('\n--------------left vs projects values-------------\n')
    print(left_vs_projects_values)

    # number of employees and number of projects done by them
    employees_and_projects = df.groupby('number_project').size()
    print('\n-------------employees and projects------------\n')
    print(employees_and_projects)

    # employees and projects values
    employees_and_projects_values = employees_and_projects.values
    print('\n--------------employees and project values--------------\n')
    print(employees_and_projects_values)

    # number of projects
    number_of_projects = np.sort(df.number_project.unique())
    print('\n--------------Number of project values--------------\n')
    print(number_of_projects)

    # bar plot of number of projects vs total number of employees in each category
    w = 0.3
    plot1 = pplot.bar(np.arange(0,len(number_of_projects)) - w/2, employees_and_projects, width=w, align='center')
    plot2 = pplot.bar(np.arange(0,len(number_of_projects)) + w/2, left_vs_projects_values, width=w, align='center')
    pplot.xticks(np.arange(0, len(number_of_projects)), number_of_projects)
    pplot.legend((plot1[0], plot2[0]), ('Total', 'Left'), loc='upper left')
    pplot.title('Total and left employees')
    pplot.xlabel('Number of projects')
    pplot.ylabel('Number of employees')
    pplot.show()

    # correlation matrix
    cm_df = df.copy()
    del cm_df['department']
    del cm_df['salary']
    cor_mat = cm_df.corr()
    hm = sns.heatmap(cor_mat)
    sns.plt.xticks(rotation='vertical')
    sns.plt.yticks(rotation='horizontal')
    sns.plt.title('Correlation matrix heatmap')
    sns.plt.tight_layout()
    sns.plt.show()


main()
