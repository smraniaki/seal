import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")

class FEACode2D:
    def __init__(self, master):
        self.master = master
        master.title("2D FEA PoC with Primitive Geometries")

        # Input frame
        self.input_frame = ttk.LabelFrame(master, text="Input Parameters")
        self.input_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        # Material properties
        ttk.Label(self.input_frame, text="Young's Modulus (Pa):").grid(row=0, column=0, sticky=tk.W)
        self.E_entry = ttk.Entry(self.input_frame)
        self.E_entry.insert(0, "1e6")
        self.E_entry.grid(row=0, column=1, pady=5)

        ttk.Label(self.input_frame, text="Poisson's Ratio:").grid(row=1, column=0, sticky=tk.W)
        self.nu_entry = ttk.Entry(self.input_frame)
        self.nu_entry.insert(0, "0.3")
        self.nu_entry.grid(row=1, column=1, pady=5)

        # Geometry
        ttk.Label(self.input_frame, text="Length (L):").grid(row=2, column=0, sticky=tk.W)
        self.L_entry = ttk.Entry(self.input_frame)
        self.L_entry.insert(0, "0.1")
        self.L_entry.grid(row=2, column=1, pady=5)

        ttk.Label(self.input_frame, text="Height (H):").grid(row=3, column=0, sticky=tk.W)
        self.H_entry = ttk.Entry(self.input_frame)
        self.H_entry.insert(0, "0.1")
        self.H_entry.grid(row=3, column=1, pady=5)

        # Geometry selection
        ttk.Label(self.input_frame, text="Geometry:").grid(row=4, column=0, sticky=tk.W)
        self.shape_var = tk.StringVar()
        self.shape_cb = ttk.Combobox(self.input_frame, textvariable=self.shape_var, values=["Square", "Circle", "Triangle"])
        self.shape_cb.current(0)
        self.shape_cb.grid(row=4, column=1, pady=5)

        # Load
        ttk.Label(self.input_frame, text="Applied Vertical Load (N):").grid(row=5, column=0, sticky=tk.W)
        self.load_entry = ttk.Entry(self.input_frame)
        self.load_entry.insert(0, "1000")
        self.load_entry.grid(row=5, column=1, pady=5)

        # Mesh resolution
        ttk.Label(self.input_frame, text="Elements in Length:").grid(row=6, column=0, sticky=tk.W)
        self.nx_entry = ttk.Entry(self.input_frame)
        self.nx_entry.insert(0, "20")
        self.nx_entry.grid(row=6, column=1, pady=5)

        ttk.Label(self.input_frame, text="Elements in Height:").grid(row=7, column=0, sticky=tk.W)
        self.ny_entry = ttk.Entry(self.input_frame)
        self.ny_entry.insert(0, "20")
        self.ny_entry.grid(row=7, column=1, pady=5)

        # Solve button
        self.solve_button = ttk.Button(self.input_frame, text="Run Analysis", command=self.run_analysis)
        self.solve_button.grid(row=8, column=0, columnspan=2, pady=10)

        # Figure for results
        self.fig, self.ax = plt.subplots(figsize=(6,4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

    def point_in_triangle(self, pt, v1, v2, v3):
        # Check if pt is inside triangle formed by v1,v2,v3 using barycentric coords
        x, y = pt
        x1,y1 = v1; x2,y2=v2; x3,y3=v3
        denom = (y2 - y3)*(x1 - x3)+(x3 - x2)*(y1 - y3)
        a = ((y2 - y3)*(x - x3)+(x3 - x2)*(y - y3))/denom
        b = ((y3 - y1)*(x - x3)+(x1 - x3)*(y - y3))/denom
        c = 1 - a - b
        return (a >=0) and (b>=0) and (c>=0)

    def run_analysis(self):
        E = float(self.E_entry.get())
        nu = float(self.nu_entry.get())
        L = float(self.L_entry.get())
        H = float(self.H_entry.get())
        load = float(self.load_entry.get())
        nx = int(self.nx_entry.get())
        ny = int(self.ny_entry.get())
        shape = self.shape_var.get()

        # Generate mesh (rectangular background)
        x_coords = np.linspace(0, L, nx+1)
        y_coords = np.linspace(0, H, ny+1)
        nnodes = (nx+1)*(ny+1)

        X = np.zeros((nnodes,2))
        count = 0
        for j in range(ny+1):
            for i in range(nx+1):
                X[count,0] = x_coords[i]
                X[count,1] = y_coords[j]
                count += 1

        # Elements (quad)
        nelem = nx*ny
        elems = np.zeros((nelem,4), dtype=int)
        count = 0
        for j in range(ny):
            for i in range(nx):
                n1 = j*(nx+1)+i
                n2 = n1+1
                n3 = n1+nx+1
                n4 = n3+1
                elems[count,:] = [n1,n2,n4,n3]
                count+=1

        # Determine which elements are inside the chosen geometry
        # We'll use element centroids for a quick inside-test
        elem_centers = np.zeros((nelem,2))
        for e in range(nelem):
            en = elems[e,:]
            elem_centers[e,:] = np.mean(X[en,:], axis=0)

        # Define geometry
        if shape == "Square":
            # Entire domain
            inside = np.ones(nelem, dtype=bool)
        elif shape == "Circle":
            # Circle centered at (L/2, H/2) with radius = min(L,H)/2
            cx, cy = L/2, H/2
            r = min(L,H)/2
            inside = ((elem_centers[:,0]-cx)**2 + (elem_centers[:,1]-cy)**2) <= r*r
        elif shape == "Triangle":
            # Right triangle with vertices (0,0), (L,0), (0,H)
            v1 = (0,0)
            v2 = (L,0)
            v3 = (0,H)
            inside = np.array([self.point_in_triangle(c, v1, v2, v3) for c in elem_centers])
        else:
            inside = np.ones(nelem, dtype=bool)

        # Filter elements
        elems = elems[inside,:]
        nelem = elems.shape[0]

        # Extract unique nodes used by these elements
        used_nodes = np.unique(elems)
        # Re-map nodes to a continuous numbering
        old_to_new = -1*np.ones(nnodes, dtype=int)
        for idx,nd in enumerate(used_nodes):
            old_to_new[nd] = idx
        X = X[used_nodes,:]
        elems = np.array([[old_to_new[n] for n in e] for e in elems])
        nnodes = X.shape[0]

        # Material parameters for plane strain
        lam = E*nu/((1+nu)*(1-2*nu))
        mu = E/(2*(1+nu))

        # Stiffness and load
        dof = 2*nnodes
        K = np.zeros((dof,dof))
        F = np.zeros(dof)

        # Integration point (single point at center)
        gauss_xi = 0.0
        gauss_eta = 0.0
        w = 2.0*2.0 # integration weight = 4 for a single-point (approx)

        N = 0.25*np.array([(1-gauss_xi)*(1-gauss_eta),
                           (1+gauss_xi)*(1-gauss_eta),
                           (1+gauss_xi)*(1+gauss_eta),
                           (1-gauss_xi)*(1+gauss_eta)])
        dN_dxi = 0.25*np.array([[-(1-gauss_eta), -(1-gauss_xi)],
                                [ (1-gauss_eta), -(1+gauss_xi)],
                                [ (1+gauss_eta),  (1+gauss_xi)],
                                [-(1+gauss_eta),  (1-gauss_xi)]])
        C = np.array([[lam+2*mu, lam,        0],
                      [lam,       lam+2*mu,  0],
                      [0,         0,         mu]])

        # Assemble
        for e in range(nelem):
            enodes = elems[e,:]
            Xe = X[enodes,:]
            # Jacobian
            J = np.zeros((2,2))
            for a in range(4):
                J[0,0] += dN_dxi[a,0]*Xe[a,0]
                J[0,1] += dN_dxi[a,0]*Xe[a,1]
                J[1,0] += dN_dxi[a,1]*Xe[a,0]
                J[1,1] += dN_dxi[a,1]*Xe[a,1]
            detJ = np.linalg.det(J)
            invJ = np.linalg.inv(J)

            dN_dx = np.zeros((4,2))
            for a in range(4):
                dN_dx[a,:] = dN_dxi[a,:].dot(invJ)

            B = np.zeros((3,8))
            for a in range(4):
                B[0,2*a]   = dN_dx[a,0]
                B[1,2*a+1] = dN_dx[a,1]
                B[2,2*a]   = dN_dx[a,1]
                B[2,2*a+1] = dN_dx[a,0]

            Ke = B.T.dot(C).dot(B)*detJ*w
            # Assembly
            for a in range(4):
                Ia = 2*enodes[a]
                for b in range(4):
                    Ib = 2*enodes[b]
                    K[Ia:Ia+2, Ib:Ib+2] += Ke[2*a:2*a+2, 2*b:2*b+2]

        # Boundary conditions:
        # Fix bottom boundary (lowest Y in the geometry)
        min_y = np.min(X[:,1])
        bottom_nodes = np.where(np.isclose(X[:,1], min_y))[0]
        fixed_dofs = []
        for n in bottom_nodes:
            fixed_dofs.append(2*n)
            fixed_dofs.append(2*n+1)

        # Apply load at top boundary (highest Y)
        max_y = np.max(X[:,1])
        top_nodes = np.where(np.isclose(X[:,1], max_y))[0]
        load_per_node = load/len(top_nodes)
        for n in top_nodes:
            F[2*n+1] -= load_per_node

        # Solve
        free_dofs = np.setdiff1d(np.arange(dof), fixed_dofs)
        Kff = K[np.ix_(free_dofs, free_dofs)]
        Ff = F[free_dofs]
        U = np.zeros(dof)
        if free_dofs.size > 0:
            U[free_dofs] = np.linalg.solve(Kff, Ff)

        UX = U[0::2]
        UY = U[1::2]

        # Plot
        self.ax.clear()
        scale = 1.0
        for e in range(nelem):
            en = elems[e,:]
            x_und = X[en,0]
            y_und = X[en,1]
            x_def = x_und + UX[en]*scale
            y_def = y_und + UY[en]*scale
            x_und = np.append(x_und, x_und[0])
            y_und = np.append(y_und, y_und[0])
            x_def = np.append(x_def, x_def[0])
            y_def = np.append(y_def, y_def[0])

            self.ax.plot(x_und, y_und, 'k--', alpha=0.3)
            self.ax.plot(x_def, y_def, 'r-')

        self.ax.set_aspect('equal', 'box')
        self.ax.set_title(f"{shape} Geometry: Deformed (red) vs Undeformed (dashed)")
        self.ax.set_xlabel("X [m]")
        self.ax.set_ylabel("Y [m]")
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = FEACode2D(root)
    # Add the trademark label at the bottom of the GUI
    trademark_label = tk.Label(root, text="@Dr. Reza Attarzadeh", font=("Arial", 8), fg="gray")
    trademark_label.pack(side=tk.BOTTOM, pady=5)
    root.mainloop()
