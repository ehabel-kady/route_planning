# Import the required Libraries
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from utils import *


def process_destination():
    # Get the values from the entry boxes
    p1 = p1.get()
    p2 = p2.get()
    share = share.get()
    if share.lower() == "yes":
        share = True
    else:
        share = False
    print(p1)
    print(p2)
    print(share)
    # Create a simple plot using matplotlib
    route_calculator = BikeRouter("LI_rattaringluse_parklad_avaandmed.csv")
    route_calculator.preprocess()
    route_calculator.calculate_shortest_path()
    p1 = "Narva mnt. 18, Tartu, Estonia"
    p2 = "Tehase 21, Tartu, Estonia"
    r_before, route, r_after = route_calculator.calculate_route(p1, p2, share)
    route_calculator.plot_routes([r_before, route, r_after])
    # Display the plot on a Tkinter window
    canvas = FigureCanvasTkAgg(plt.gcf(), master=root)
    canvas.draw()
    canvas.get_tk_widget().pack()


# Create the main window
root = tk.Tk()
root.title("Route calculator")
root.geometry("750x750")
# Create three entry boxes
label_p1 = tk.Label(root, text="Source:")
label_p1.pack()
p1 = tk.Entry(root)
p1.pack()

label_p2 = tk.Label(root, text="Destination:")
label_p2.pack()
p2 = tk.Entry(root)
p2.pack()

label_share = tk.Label(root, text="will you use bike share? (yes/no)")
label_share.pack()
share = tk.Entry(root)
share.pack()
# Button to plot the graph
plot_button = tk.Button(root, text="Get Route", command=process_destination)
plot_button.pack()

# Run the Tkinter main loop
root.mainloop()
