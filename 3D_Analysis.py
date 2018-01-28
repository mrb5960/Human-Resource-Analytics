import pandas as pd
import matplotlib.pyplot as pplot
from mpl_toolkits.mplot3d import Axes3D


def main():
    path = "C:\\Study\\RIT\\BDA\\BDA_Term_Project\\HR_comma_sep.csv"

    df = pd.read_csv(path)
    df_new = df.copy()
    departments = df_new.department.unique().tolist()
    #categorical to numneric
    df_new['department'] = df_new['department'].replace(to_replace=departments, value=range(0,len(departments)))
    #print(df_new['department'].unique())
    salaries = df_new.salary.unique().tolist()
    # categorical to numneric
    df_new['salary'] = df_new['salary'].replace(to_replace=salaries, value=range(0, len(salaries)))


    fig = pplot.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = df_new['satisfaction_level']
    y = df_new['average_monthly_hours']
    z = df_new['last_evaluation']
    c = df_new['left']
    _ = ax.scatter(xs=x, ys=y, zs=z, c=c)
    _ = ax.set_xlabel('Satisfaction level')
    _ = ax.set_ylabel('Average monthly hours')
    _ = ax.set_zlabel('Last evaluation')
    _ = pplot.title('3D visualization')
    pplot.show()

main()
