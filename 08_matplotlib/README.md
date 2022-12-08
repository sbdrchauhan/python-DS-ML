## Install matplotlib using `pip`:
```bash
pip install matplotlib              # to install for the first time
pip install --upgrade matplotlib    # to upgrade already installed ones
```

## Import package:
```python
import matplotlib.pyplot as plt
```

## Style of the Plot:
```python
plt.style.available                # to see what are available styles to choose from
plt.style.use('fivethirtyeight')   # to use one of available styles
```

## 01. Line Plot (Basic):
```python
# developer ages (x values)
dev_age = [25,26,27,28,29,30,31,32,33,34,35]   # list of ages
# median salaries (y values)
dev_salaries = [38496, 42000, 46752, 49320, 53200, 
            56000, 62316, 64928, 67317, 68748, 73752]

# plot it. 'label' argument helps to make legend later
# format strings
# fmt = '[marker][line][color]'. '--k' dash line color black
plt.plot(dev_age, dev_salaries, '--k' label='All Devs')
# same as above but format more explicit
plt.plot(dev_age, dev_salaries, color='k', linestyle='--', marker='*', linewidth=3, label='All Devs')

# now look for python dev
# Median Python Developer Salaries by Age
# same x-range in both plots, could use one
py_dev_x = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
py_dev_y = [45372, 48876, 53850, 57287, 63016,
            65998, 70003, 70000, 71496, 75370, 83640]

# plotting two lines in one plot
# color can be hex color codes: e.g. #5a7d9a, can find online for these
plt.plot(py_dev_x, py_dev_y, color='r', linestyle='-.', marker='o', label='Python Devs')

# add labels to axis and also title
plt.title("Median Salary (USD) by Age")
plt.xlabel('Ages')
plt.ylabel('Median Salary (USD)')

# add legend into the plot
plt.legend()

# to show gridlines
plt.grid(True)

# padding need sometime
plt.tight_layout()

# to save fig
plt.savefig('plot.png')

# to show the plot
plt.show()
```
The final output plot from above code comes out something like this.
![basiclineplot](./plot1.png)