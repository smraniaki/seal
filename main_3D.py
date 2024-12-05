import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

matplotlib.use("TkAgg")

class FEACode3D:
    def __init__(self, master):
        self.master = master
        master.title("3D FEA PoC for Elastomeric Seal Frame")

        # Input frame
        self.input_frame = ttk.LabelFrame(master, text="Input Parameters")
        self.input_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        # Material
        ttk.Label(self.input_frame, text="Young's Modulus (Pa):").grid(row=0, column=0, sticky=tk.W)
        self.E_entry = ttk.Entry(self.input_frame)
        self.E_entry.insert(0, "1e6")
        self.E_entry.grid(row=0, column=1, pady=5)

        ttk.Label(self.input_frame, text="Poisson's Ratio:").grid(row=1, column=0, sticky=tk.W)
        self.nu_entry = ttk.Entry(self.input_frame)
        self.nu_entry.insert(0, "0.45")
        self.nu_entry.grid(row=1, column=1, pady=5)

        # Geometry
        ttk.Label(self.input_frame, text="Length (X) [m]:").grid(row=2, column=0, sticky=tk.W)
        self.L_entry = ttk.Entry(self.input_frame)
        self.L_entry.insert(0, "0.1")
        self.L_entry.grid(row=2, column=1, pady=5)

        ttk.Label(self.input_frame, text="Width (Y) [m]:").grid(row=3, column=0, sticky=tk.W)
        self.W_entry = ttk.Entry(self.input_frame)
        self.W_entry.insert(0, "0.05")
        self.W_entry.grid(row=3, column=1, pady=5)

        ttk.Label(self.input_frame, text="Height (Z) [m]:").grid(row=4, column=0, sticky=tk.W)
        self.H_entry = ttk.Entry(self.input_frame)
        self.H_entry.insert(0, "0.02")
        self.H_entry.grid(row=4, column=1, pady=5)

        # Load
        ttk.Label(self.input_frame, text="Applied Vertical Load (N):").grid(row=5, column=0, sticky=tk.W)
        self.load_entry = ttk.Entry(self.input_frame)
        self.load_entry.insert(0, "1000")
        self.load_entry.grid(row=5, column=1, pady=5)

        # Mesh resolution
        ttk.Label(self.input_frame, text="Elements in X:").grid(row=6, column=0, sticky=tk.W)
        self.nx_entry = ttk.Entry(self.input_frame)
        self.nx_entry.insert(0, "4")
        self.nx_entry.grid(row=6, column=1, pady=5)

        ttk.Label(self.input_frame, text="Elements in Y:").grid(row=7, column=0, sticky=tk.W)
        self.ny_entry = ttk.Entry(self.input_frame)
        self.ny_entry.insert(0, "2")
        self.ny_entry.grid(row=7, column=1, pady=5)

        ttk.Label(self.input_frame, text="Elements in Z:").grid(row=8, column=0, sticky=tk.W)
        self.nz_entry = ttk.Entry(self.input_frame)
        self.nz_entry.insert(0, "2")
        self.nz_entry.grid(row=8, column=1, pady=5)

        # Solve button
        self.solve_button = ttk.Button(self.input_frame, text="Run Analysis", command=self.run_analysis)
        self.solve_button.grid(row=9, column=0, columnspan=2, pady=10)

        # Figure for results
        self.fig = plt.Figure(figsize=(6,4))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

    def run_analysis(self):
        E = float(self.E_entry.get())
        nu = float(self.nu_entry.get())
        L = float(self.L_entry.get())
        W = float(self.W_entry.get())
        H = float(self.H_entry.get())
        load = float(self.load_entry.get())
        nx = int(self.nx_entry.get())
        ny = int(self.ny_entry.get())
        nz = int(self.nz_entry.get())

        # Generate mesh
        x_coords = np.linspace(0, L, nx+1)
        y_coords = np.linspace(0, W, ny+1)
        z_coords = np.linspace(0, H, nz+1)

        nnodes = (nx+1)*(ny+1)*(nz+1)
        X = np.zeros((nnodes, 3))
        count = 0
        for k in range(nz+1):
            for j in range(ny+1):
                for i in range(nx+1):
                    X[count,0] = x_coords[i]
                    X[count,1] = y_coords[j]
                    X[count,2] = z_coords[k]
                    count += 1

        # Elements: 8-node bricks
        nelem = nx*ny*nz
        elems = np.zeros((nelem,8), dtype=int)
        count = 0
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    n000 = k*(ny+1)*(nx+1) + j*(nx+1) + i
                    n100 = n000 + 1
                    n010 = n000 + (nx+1)
                    n110 = n010 + 1
                    n001 = n000 + (ny+1)*(nx+1)
                    n101 = n001 + 1
                    n011 = n001 + (nx+1)
                    n111 = n011 + 1
                    elems[count,:] = [n000, n100, n110, n010, n001, n101, n111, n011]
                    count+=1

        # Material parameters for 3D elasticity
        lam = E*nu/((1+nu)*(1-2*nu))
        mu = E/(2*(1+nu))

        # Constitutive matrix for 3D
        # [exx eyy ezz exy eyz ezx]^T
        # C is 6x6
        C = np.array([
            [lam+2*mu, lam,       lam,       0,    0,    0],
            [lam,       lam+2*mu, lam,       0,    0,    0],
            [lam,       lam,       lam+2*mu, 0,    0,    0],
            [0,         0,         0,        mu,   0,    0],
            [0,         0,         0,        0,    mu,   0],
            [0,         0,         0,        0,    0,    mu]
        ])

        # DOF = 3 * nnodes
        dof = 3*nnodes
        K = np.zeros((dof,dof))
        F = np.zeros(dof)

        # Single integration point at center of element: xi, eta, zeta = 0,0,0
        # Shape functions for an 8-node brick at (0,0,0):
        # N_i = 1/8(1±ξ)(1±η)(1±ζ)
        # For ξ,η,ζ = 0:
        # All N = 1/8
        N = np.ones(8)*0.125

        # Derivatives w.r.t local coords (dN/dξ, dN/dη, dN/dζ) at center:
        # Each node differs by sign pattern:
        # Node ordering assumed as:
        # (Following standard Hex Node numbering)
        #    Bottom face (z=0): 
        #        n000 ( - - - ), n100 (+ - -), n110 (+ + -), n010 (- + -)
        #    Top face (z=1):
        #        n001 (- - +), n101 (+ - +), n111 (+ + +), n011 (- + +)
        # With ξ,η,ζ in [-1,1]
        dN_dxi = np.array([
            [ -1, -1, -1],
            [  1, -1, -1],
            [  1,  1, -1],
            [ -1,  1, -1],
            [ -1, -1,  1],
            [  1, -1,  1],
            [  1,  1,  1],
            [ -1,  1,  1]
        ])*0.125

        # Assemble K
        for e in range(nelem):
            enodes = elems[e,:]
            Xe = X[enodes,:]  # 8x3
            # Compute Jacobian
            # J = dX/dξ at center (3x3)
            J = np.zeros((3,3))
            for a in range(8):
                J[0,0] += dN_dxi[a,0]*Xe[a,0]
                J[0,1] += dN_dxi[a,0]*Xe[a,1]
                J[0,2] += dN_dxi[a,0]*Xe[a,2]
                J[1,0] += dN_dxi[a,1]*Xe[a,0]
                J[1,1] += dN_dxi[a,1]*Xe[a,1]
                J[1,2] += dN_dxi[a,1]*Xe[a,2]
                J[2,0] += dN_dxi[a,2]*Xe[a,0]
                J[2,1] += dN_dxi[a,2]*Xe[a,1]
                J[2,2] += dN_dxi[a,2]*Xe[a,2]

            detJ = np.linalg.det(J)
            invJ = np.linalg.inv(J)

            # dN_dx = dN_dξ * invJ
            dN_dx = np.zeros((8,3))
            for a in range(8):
                dN_dx[a,:] = dN_dxi[a,:].dot(invJ)

            # Construct B matrix (6x24 for 8 nodes)
            B = np.zeros((6,24))
            for a in range(8):
                ix = 3*a
                dNx = dN_dx[a,0]
                dNy = dN_dx[a,1]
                dNz = dN_dx[a,2]
                B[0,ix  ] = dNx
                B[1,ix+1] = dNy
                B[2,ix+2] = dNz
                B[3,ix  ] = dNy
                B[3,ix+1] = dNx
                B[4,ix+1] = dNz
                B[4,ix+2] = dNy
                B[5,ix  ] = dNz
                B[5,ix+2] = dNx

            # Single point integration weight: volume * 1 gauss point
            # Actual integration would be sum over gauss points; here just one:
            w = 2.0 * 2.0 * 2.0  # For a single point at center of [-1,1]^3, 
                                 # scale factor for full domain is 8. 
                                 # This is a gross simplification for PoC.
            Ke = B.T @ C @ B * detJ * (w/8.0) # dividing by 8 to partially correct overshoot

            # Assemble to global K
            for a in range(8):
                Ia = 3*enodes[a]
                for b in range(8):
                    Ib = 3*enodes[b]
                    K[Ia:Ia+3, Ib:Ib+3] += Ke[3*a:3*a+3, 3*b:3*b+3]

        # Boundary conditions:
        # Fix bottom face Z=0 in all directions
        bottom_nodes = [n for n in range(nnodes) if X[n,2] == 0.0]
        fixed_dofs = []
        for n in bottom_nodes:
            fixed_dofs.append(3*n)
            fixed_dofs.append(3*n+1)
            fixed_dofs.append(3*n+2)

        # Apply load on top face (Z=H) in negative Z-direction
        top_nodes = [n for n in range(nnodes) if X[n,2] == H]
        load_per_node = load/len(top_nodes)
        for n in top_nodes:
            F[3*n+2] -= load_per_node

        # Solve system
        all_dofs = np.arange(dof)
        free_dofs = np.setdiff1d(all_dofs, fixed_dofs)

        Kff = K[np.ix_(free_dofs, free_dofs)]
        Ff = F[free_dofs]

        U = np.zeros(dof)
        U[free_dofs] = np.linalg.solve(Kff, Ff)

        UX = U[0::3]
        UY = U[1::3]
        UZ = U[2::3]

        # Visualization
        self.ax.clear()
        # Plot a subset of edges of the mesh: 
        # We'll plot the edges of each element in undeformed (light) and deformed (dark)
        scale = 1.0  # Scale factor for deformed shape

        # Define element edges (each hexa element has 12 edges)
        edges = [
            [0,1], [1,2], [2,3], [3,0],       # bottom face
            [4,5], [5,6], [6,7], [7,4],       # top face
            [0,4], [1,5], [2,6], [3,7]        # vertical edges
        ]

        for e in range(nelem):
            en = elems[e,:]
            # Undeformed
            Xe = X[en,:]
            # Deformed
            Xd = np.copy(Xe)
            Xd[:,0] += UX[en]*scale
            Xd[:,1] += UY[en]*scale
            Xd[:,2] += UZ[en]*scale

            for edge in edges:
                n1, n2 = edge
                self.ax.plot([Xe[n1,0], Xe[n2,0]], [Xe[n1,1], Xe[n2,1]], [Xe[n1,2], Xe[n2,2]], 
                             color='gray', linestyle='--', alpha=0.3)
                self.ax.plot([Xd[n1,0], Xd[n2,0]], [Xd[n1,1], Xd[n2,1]], [Xd[n1,2], Xd[n2,2]], 
                             color='red')

        self.ax.set_box_aspect([L, W, H])  # Make aspect ratio close to real
        self.ax.set_xlabel('X [m]')
        self.ax.set_ylabel('Y [m]')
        self.ax.set_zlabel('Z [m]')
        self.ax.set_title("3D Deformed (red) vs Undeformed (gray) Mesh")
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = FEACode3D(root)
    
    # Add the trademark label at the bottom of the GUI
    trademark_label = tk.Label(root, text="@Dr. Reza Attarzadeh", font=("Arial", 8), fg="gray")
    trademark_label.pack(side=tk.BOTTOM, pady=5)

    root.mainloop()
