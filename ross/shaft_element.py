import os
from pathlib import Path

import bokeh.palettes as bp
import matplotlib.patches as mpatches
import numpy as np
import toml
from bokeh.models import ColumnDataSource, HoverTool

import ross
from ross.element import Element
from ross.materials import Material, steel
from ross.utils import read_table_file

__all__ = ["ShaftElement", "ShaftTaperedElement"]

bokeh_colors = bp.RdGy[11]

class ShaftElement(Element):
    r"""A shaft element.
    This class will create a shaft element that may take into
    account shear, rotary inertia an gyroscopic effects.
    The matrices will be defined considering the following local
    coordinate vector:

    .. math::

        [x_0, y_0, \alpha_0, \beta_0, x_1, y_1, \alpha_1, \beta_1]^T
    Where :math:`\alpha_0` and :math:`\alpha_1` are the bending on the yz plane
    and :math:`\beta_0` and :math:`\beta_1` are the bending on the xz plane.

    Parameters
    ----------
    L : float
        Element length.
    i_d : float
        Inner diameter of the element.
    o_d : float
        Outer diameter of the element.
    material : ross.material
        Shaft material.
    n : int, optional
        Element number (coincident with it's first node).
        If not given, it will be set when the rotor is assembled
        according to the element's position in the list supplied to
        the rotor constructor.
    axial_force : float, optional
        Axial force.
    torque : float, optional
        Torque.
    shear_effects : bool, optional
        Determine if shear effects are taken into account.
        Default is True.
    rotary_inertia : bool, optional
        Determine if rotary_inertia effects are taken into account.
        Default is True.
    gyroscopic : bool, optional
        Determine if gyroscopic effects are taken into account.
        Default is True.
    shear_method_calc : string, optional
        Determines which shear calculation method the user will adopt
        Default is 'cowper'

    Returns
    -------

    Attributes
    ----------
    Poisson : float
        Poisson coefficient for the element.
    A : float
        Element section area.
    Ie : float
        Ie is the second moment of area of the cross section about
        the neutral plane :math:`Ie = \pi r^2/4`
    phi : float
        Constant that is used according to :cite:`friswell2010dynamics` to
        consider rotary inertia and shear effects. If these are not considered
        :math:`\phi=0`.

    References
    ----------

    .. bibliography:: ../../../docs/refs.bib

    Examples
    --------
    >>> from ross.materials import steel
    >>> le = 0.25
    >>> i_d = 0
    >>> o_d = 0.05
    >>> Euler_Bernoulli_Element = ShaftElement(le, i_d, o_d, steel,
    ...                                        shear_effects=False, rotary_inertia=False)
    >>> Euler_Bernoulli_Element.phi
    0
    >>> Timoshenko_Element = ShaftElement(le, i_d, o_d, steel,
    ...                                   rotary_inertia=True,
    ...                                   shear_effects=True)
    >>> Timoshenko_Element.phi
    0.08795566502463055
    """

    def __init__(
        self,
        L,
        i_d,
        o_d,
        material,
        n=None,
        axial_force=0,
        torque=0,
        shear_effects=True,
        rotary_inertia=True,
        gyroscopic=True,
        tag=None,
    ):

        if type(material) is str:
            os.chdir(Path(os.path.dirname(ross.__file__)))
            self.material = Material.use_material(material)
        else:
            self.material = material

        self.shear_effects = shear_effects
        self.rotary_inertia = rotary_inertia
        self.gyroscopic = gyroscopic
        self.axial_force = axial_force
        self.torque = torque
        self._n = n
        self.n_l = n
        self.n_r = None
        if n is not None:
            self.n_r = n + 1

        self.tag = tag

        self.L = float(L)

        # diameters
        self.i_d = float(i_d)
        self.o_d = float(o_d)
        self.i_d_l = float(i_d)
        self.o_d_l = float(o_d)
        self.i_d_r = float(i_d)
        self.o_d_r = float(o_d)
        self.color = self.material.color

        # Timoshenko kappa factor determination, based on the diameters relation
        if self.__is_circular():
            kappa = (6 * (1 + self.material.Poisson) ** 2) / (
                7 + 12 * self.material.Poisson + 4 * self.material.Poisson ** 2
            )
        elif self.__is_thickwall():
            a = self.i_d
            b = self.o_d
            v = self.material.Poisson
            kappa = (6(*a ** 2 + b ** 2) ** 2 * (1 + v) ** 2) / (
                7 * a ** 4
                + 34 * a ** 2 * b ** 2
                + 7 * b ** 4
                + v(12 * a ** 4 + 48 * a ** 2 * b ** 2 + 12 * b ** 4)
                + v ** 2 * (4 * a ** 4 + 16 * a ** 2 * b ** 2 + 4 * b ** 4)
            )
        else:
            kappa = (1 + self.material.Poisson) / (2 + self.material.Poisson)

        self.kappa = kappa

    def __eq__(self, other):
        """
        Equality method for comparasions

        Parameters
        ----------
        other : obj
            parameter for comparasion

        Returns
        -------
        True if other is equal to the reference parameter.
        False if not.

        Example
        -------
        >>> from ross.materials import steel
        >>> le = 0.25
        >>> i_d = 0
        >>> o_d = 0.05
        >>> shaft1 = ShaftElement(
        ...        le, i_d, o_d, steel, rotary_inertia=True, shear_effects=True
        ... )
        >>> shaft2 = ShaftElement(
        ...        le, i_d, o_d, steel, rotary_inertia=True, shear_effects=True
        ... )
        >>> shaft1 == shaft2
        True
        """
        if self.__dict__ == other.__dict__:
            return True
        else:
            return False

    def __hash__(self):
        return hash(self.tag)

    def __is_circular(self):
        return self.shear_effects and self.i_d == 0

    def __is_thickwall(self):
        p = (self.o_d - self.i_d) / self.o_d
        return self.shear_effects and p >= 0.2

    def save(self, file_name):
        """Save shaft elements to toml file.

        Parameters
        ----------
        file_name : str

        Examples
        --------
        >>> from ross.materials import steel
        >>> le = 0.25
        >>> i_d = 0
        >>> o_d = 0.05
        >>> shaft1 = ShaftElement(
        ...     le, i_d, o_d, steel, rotary_inertia=True, shear_effects=True
        ... )
        >>> shaft1.save('ShaftElement.toml')
        """
        data = self.load_data(file_name)
        data["ShaftElement"][str(self.n)] = {
            "L": self.L,
            "i_d": self.i_d,
            "o_d": self.o_d,
            "material": self.material.name,
            "n": self.n,
            "axial_force": self.axial_force,
            "torque": self.torque,
            "shear_effects": self.shear_effects,
            "rotary_inertia": self.rotary_inertia,
            "gyroscopic": self.gyroscopic,
            "shear_method_calc": self.shear_method_calc,
        }
        self.dump_data(data, file_name)

    @staticmethod
    def load(file_name="ShaftElement"):
        """Load previously saved shaft elements from toml file.

        Parameters
        ----------
        file_name : str, optional

        Examples
        --------
        >>> from ross.materials import steel
        >>> le = 0.25
        >>> i_d = 0
        >>> o_d = 0.05
        >>> shaft1 = ShaftElement(
        ...     le, i_d, o_d, steel, rotary_inertia=True, shear_effects=True
        ... )
        >>> shaft1.save('ShaftElement.toml')
        >>> shaft2 = ShaftElement.load("ShaftElement.toml")
        >>> shaft2
        [ShaftElement(L=0.25, i_d=0.0, o_d=0.05, material='Steel', n=None)]
        """
        shaft_elements = []
        with open("ShaftElement.toml", "r") as f:
            shaft_elements_dict = toml.load(f)
            for element in shaft_elements_dict["ShaftElement"]:
                shaft_elements.append(
                    ShaftElement(**shaft_elements_dict["ShaftElement"][element])
                )
        return shaft_elements

    @classmethod
    def from_table(cls, file, sheet_type="Simple", sheet_name=0):
        """Instantiate one or more shafts using inputs from an Excel table.

        A header with the names of the columns is required. These names should
        match the names expected by the routine (usually the names of the
        parameters, but also similar ones). The program will read every row
        bellow the header until they end or it reaches a NaN.

        Parameters
        ----------
        file: str
            Path to the file containing the shaft parameters.
        sheet_type: str, optional
            Describes the kind of sheet the function should expect:
                Simple: The input table should specify only the number of the materials to be used.
                They must be saved prior to calling the method.
                Model: The materials parameters must be passed along with the shaft parameters. Each
                material must have an id number and each shaft must reference one of the materials ids.
        sheet_name: int or str, optional
            Position of the sheet in the file (starting from 0) or its name. If none is passed, it is
            assumed to be the first sheet in the file.

        Returns
        -------
        shaft: list
            A list of shaft objects.
        """
        parameters = read_table_file(
            file, "shaft", sheet_name=sheet_name, sheet_type=sheet_type
        )
        list_of_shafts = []
        if sheet_type == "Model":
            new_materials = {}
            for i in range(0, len(parameters["matno"])):
                new_material = Material(
                    name="shaft_mat_" + str(parameters["matno"][i]),
                    rho=parameters["rhoa"][i],
                    E=parameters["ea"][i],
                    G_s=parameters["ga"][i],
                )
                new_materials["shaft_mat_" + str(parameters["matno"][i])] = new_material
            for i in range(0, len(parameters["L"])):
                list_of_shafts.append(
                    cls(
                        L=parameters["L"][i],
                        i_d=parameters["i_d"][i],
                        o_d=parameters["o_d"][i],
                        material=new_materials[parameters["material"][i]],
                        n=parameters["n"][i],
                        axial_force=parameters["axial_force"][i],
                        torque=parameters["torque"][i],
                        shear_effects=parameters["shear_effects"][i],
                        rotary_inertia=parameters["rotary_inertia"][i],
                        gyroscopic=parameters["gyroscopic"][i],
                        shear_method_calc=parameters["shear_method_calc"][i],
                    )
                )
        elif sheet_type == "Simple":
            for i in range(0, len(parameters["L"])):
                list_of_shafts.append(
                    cls(
                        L=parameters["L"][i],
                        i_d=parameters["i_d"][i],
                        o_d=parameters["o_d"][i],
                        material=parameters["material"][i],
                        n=parameters["n"][i],
                        axial_force=parameters["axial_force"][i],
                        torque=parameters["torque"][i],
                        shear_effects=parameters["shear_effects"][i],
                        rotary_inertia=parameters["rotary_inertia"][i],
                        gyroscopic=parameters["gyroscopic"][i],
                        shear_method_calc=parameters["shear_method_calc"][i],
                    )
                )
        return list_of_shafts

    @property
    def n(self):
        """
        Set the element number as property

        Parameters
        ----------
        Returns
        -------
        n : int
            Element number
        """
        return self._n

    @n.setter
    def n(self, value):
        """
        Method to set a new value for the element number.

        Parameters
        ----------
        value : int
            element number

        Returns
        -------
        Examples
        --------
        >>> from ross.materials import steel
        >>> shaft1 = ShaftElement(L=0.25, i_d=0, o_d=0.05, material=steel,
        ...                       rotary_inertia=True, shear_effects=True)
        >>> shaft1.n = 0
        >>> shaft1
        ShaftElement(L=0.25, i_d=0.0, o_d=0.05, material='Steel', n=0)
        """
        self._n = value
        self.n_l = value
        if value is not None:
            self.n_r = value + 1

    def dof_mapping(self):
        """
        Method to map the element's degrees of freedom

        Parameters
        ----------
        Returns
        -------
        The numbering of degrees of freedom of each element node.
        """
        return dict(
            x_0=0, y_0=1, alpha_0=2, beta_0=3, x_1=4, y_1=5, alpha_1=6, beta_1=7
        )

    def __repr__(self):
        """This function returns a string representation of a shaft element.

        Parameters
        ----------
        Returns
        -------
        A string representation of a shaft object.

        Examples
        --------
        >>> from ross.materials import steel
        >>> shaft1 = ShaftElement(L=0.25, i_d=0, o_d=0.05, material=steel,
        ...                       rotary_inertia=True, shear_effects=True)
        >>> shaft1
        ShaftElement(L=0.25, i_d=0.0, o_d=0.05, material='Steel', n=None)
        """
        return (
            f"{self.__class__.__name__}"
            f"(L={self.L:{0}.{5}}, i_d={self.i_d:{0}.{5}}, "
            f"o_d={self.o_d:{0}.{5}}, material={self.material.name!r}, "
            f"n={self.n})"
        )

    def __str__(self):
        """
        Method to convert object into string
        
        Parameters
        ----------
        Returns
        -------
        The object's parameters translated to strings
        """
        return (
            f"\nElem. N:    {self.n}"
            f"\nLenght:     {self.L:{10}.{5}}"
            f"\nInt. Diam.: {self.i_d:{10}.{5}}"
            f"\nOut. Diam.: {self.o_d:{10}.{5}}"
            f'\n{35*"-"}'
            f"\n{self.material}"
            f"\n"
        )

    def M(self):
        r"""Mass matrix for an instance of a shaft element.

        Returns
        -------
        M: np.ndarray
            Mass matrix for the shaft element.

        Examples
        --------
        >>> Timoshenko_Element = ShaftElement(0.25, 0, 0.05, steel,
        ...                                  rotary_inertia=True,
        ...                                  shear_effects=True)
        >>> Timoshenko_Element.M()[:4, :4]
        array([[ 1.42050794,  0.        ,  0.        ,  0.04931719],
               [ 0.        ,  1.42050794, -0.04931719,  0.        ],
               [ 0.        , -0.04931719,  0.00231392,  0.        ],
               [ 0.04931719,  0.        ,  0.        ,  0.00231392]])
        """
        
        # temporary material and geometrical constants
        L = self.L
        tempG = self.material.E / (2 * (1 + self.n))
        tempS = np.pi * ((self.o_d / 2) ** 2 - (self.i_d / 2) ** 2)
        tempI = np.pi / 4 * ((self.o_d / 2) ** 4 - (self.i_d / 2) ** 4)
        tempJ = np.pi / 2 * ((self.o_d / 2) ** 4 - (self.i_d / 2) ** 4)
        tempMM = (self.i_d / 2) / (self.o_d / 2)

        # temporary variables dependent on kappa
        tempA = (
            12
            * self.material.E
            * tempI
            / (self.material.G_s * self.kappa * tempS * L ** 2)
        )

        # element level matrix declaration

        aux1 = self.material.rho * tempS * L / 420
        aux2 = self.material.rho * tempJ * L / 6
        # fmt: off
        M=aux1*np.array([
            [156        ,0           ,0           ,0           ,0           ,-22*L       ,54          ,0           ,0           ,0           ,0           ,13*L  ]   
            [0          ,140         ,0           ,0           ,0           ,0           ,0           ,70          ,0           ,0           ,0           ,0     ]    
            [0          ,0           ,156         ,22*L        ,0           ,0           ,0           ,0           ,54          ,-13*L       ,0           ,0     ]    
            [0          ,0           ,22*L        ,4*L^2       ,0           ,0           ,0           ,0           ,13*L        ,-3*L^2      ,0           ,0     ]     
            [0          ,0           ,0           ,0           ,2*aux2/aux1 ,0           ,0           ,0           ,0           ,0           ,aux2/aux1   ,0     ]    
            [-22*L      ,0           ,0           ,0           ,0           ,4*L^2       ,-13*L       ,0           ,0           ,0           ,0           ,-3*L^2]    
            [54         ,0           ,0           ,0           ,0           ,-13*L       ,156         ,0           ,0           ,0           ,0           ,22*L  ]      
            [0          ,70          ,0           ,0           ,0           ,0           ,0           ,140         ,0           ,0           ,0           ,0     ]      
            [0          ,0           ,54          ,13*L        ,0           ,0           ,0           ,0           ,156         ,-22*L       ,0           ,0     ]     
            [0          ,0           ,-13*L       ,-3*L^2      ,0           ,0           ,0           ,0           ,-22*L       ,4*L^2       ,0           ,0     ] 
            [0          ,0           ,0           ,0           ,aux2/aux1   ,0           ,0           ,0           ,0           ,0           ,2*aux2/aux1 ,0     ]
            [13*L       ,0           ,0           ,0           ,0           ,-3*L^2      ,22*L        ,0           ,0           ,0           ,0           ,4*L^2 ]
        ])

        Ms=self.material.rho*tempI/(30*L)*np.array([
            [36         ,0           ,0           ,0           ,0           ,-3*L        ,-36         ,0           ,0           ,0           ,0           ,-3*L ]    
            [0          ,0           ,0           ,0           ,0           ,0           ,0           ,0           ,0           ,0           ,0           ,0    ]     
            [0          ,0           ,36          ,3*L         ,0           ,0           ,0           ,0           ,-36         ,3*L         ,0           ,0    ]    
            [0          ,0           ,3*L         ,4*L^2       ,0           ,0           ,0           ,0           ,-3*L        ,-L^2        ,0           ,0    ]      
            [0          ,0           ,0           ,0           ,0           ,0           ,0           ,0           ,0           ,0           ,0           ,0    ]      
            [-3*L       ,0           ,0           ,0           ,0           ,4*L^2       ,3*L         ,0           ,0           ,0           ,0           ,-L^2 ]     
            [-36        ,0           ,0           ,0           ,0           ,3*L         ,36          ,0           ,0           ,0           ,0           ,3*L  ]   
            [0          ,0           ,0           ,0           ,0           ,0           ,0           ,0           ,0           ,0           ,0           ,0    ] 
            [0          ,0           ,-36         ,-3*L        ,0           ,0           ,0           ,0           ,36          ,-3*L        ,0           ,0    ]     
            [0          ,0           ,3*L         ,-L^2        ,0           ,0           ,0           ,0           ,-3*L        ,4*L^2       ,0           ,0    ]       
            [0          ,0           ,0           ,0           ,0           ,0           ,0           ,0           ,0           ,0           ,0           ,0    ]     
            [-3*L       ,0           ,0           ,0           ,0           ,-L^2        ,3*L         ,0           ,0           ,0           ,0           ,4*L^2]
        ])
        # fmt: on
        
        M = M + Ms

        return M

    def K(self):
        r"""Stiffness matrix for an instance of a shaft element.

        Returns
        -------
        K: np.ndarray
            Stiffness matrix for the shaft element.

        Examples
        --------
        >>> from ross.materials import steel
        >>> Timoshenko_Element = ShaftElement(0.25, 0, 0.05, steel,
        ...                                  rotary_inertia=True,
        ...                                  shear_effects=True)
        >>> Timoshenko_Element.K()[:4, :4]/1e6
        array([[45.69644273,  0.        ,  0.        ,  5.71205534],
               [ 0.        , 45.69644273, -5.71205534,  0.        ],
               [ 0.        , -5.71205534,  0.97294287,  0.        ],
               [ 5.71205534,  0.        ,  0.        ,  0.97294287]])
        """

        # temporary material and geometrical constants
        L = self.L
        tempG = self.material.E / (2 * (1 + self.n))
        tempS = np.pi * ((self.o_d / 2) ** 2 - (self.i_d / 2) ** 2)
        tempI = np.pi / 4 * ((self.o_d / 2) ** 4 - (self.i_d / 2) ** 4)
        tempJ = np.pi / 2 * ((self.o_d / 2) ** 4 - (self.i_d / 2) ** 4)
        tempMM = (self.i_d / 2) / (self.o_d / 2)

        # temporary variables dependent on kappa
        tempA = (
            12
            * self.material.E
            * tempI
            / (self.material.G_s * self.kappa * tempS * L ** 2)
        )

        # element level matrix declaration

        aux1 = self.material.E * tempI / ((1 + tempA) * L**3)
        aux2 = self.material.G * tempJ / L
        aux3 = self.material.E * tempS / L

        KcEst = aux1*np.array([12   ,0          ,0   ,0         ,0          ,-6*L      ,-12 ,0          ,0    ,0         ,0          ,-6*L     ]  
            [0    ,aux3/aux1  ,0   ,0         ,0          ,0         ,0   ,-aux3/aux1 ,0    ,0         ,0          ,0        ]
            [0    ,0          ,12  ,6*L       ,0          ,0         ,0   ,0          ,-12  ,6*L       ,0          ,0        ]  
            [0    ,0          ,6*L ,(4+a)*L^2 ,0          ,0         ,0   ,0          ,-6*L ,(2-a)*L^2 ,0          ,0        ]  
            [0    ,0          ,0   ,0         ,aux2/aux1  ,0         ,0   ,0          ,0    ,0         ,-aux2/aux1 ,0        ] 
            [-6*L ,0          ,0   ,0         ,0          ,(4+a)*L^2 ,6*L ,0          ,0    ,0         ,0          ,(2-a)*L^2]   
            [-12  ,0          ,0   ,0         ,0          ,6*L       ,12  ,0          ,0    ,0         ,0          ,6*L      ]  
            [0    ,-aux3/aux1 ,0   ,0         ,0          ,0         ,0   ,aux3/aux1  ,0    ,0         ,0          ,0        ]  
            [0    ,0          ,-12 ,-6*L      ,0          ,0         ,0   ,0          ,12   ,-6*L      ,0          ,0        ]   
            [0    ,0          ,6*L ,(2-a)*L^2 ,0          ,0         ,0   ,0          ,-6*L ,(4+a)*L^2 ,0          ,0        ]  
            [0    ,0          ,0   ,0         ,-aux2/aux1 ,0         ,0   ,0          ,0    ,0         ,aux2/aux1  ,0        ]
            [-6*L ,0          ,0   ,0         ,0          ,(2-a)*L^2 ,6*L ,0          ,0    ,0         ,0          ,(4+a)*L^2])
    
        Kstart = self.material.rho*tempI/(15*L)*np.array([0 ,0 ,-36 ,-3*L  ,0 ,0 ,0 ,0 ,36   ,-3*L  ,0 ,0]
            [0 ,0 ,0   ,0     ,0 ,0 ,0 ,0 ,0    ,0     ,0 ,0]
            [0 ,0 ,0   ,0     ,0 ,0 ,0 ,0 ,0    ,0     ,0 ,0]
            [0 ,0 ,0   ,0     ,0 ,0 ,0 ,0 ,0    ,0     ,0 ,0]
            [0 ,0 ,0   ,0     ,0 ,0 ,0 ,0 ,0    ,0     ,0 ,0]
            [0 ,0 ,3*L ,4*L^2 ,0 ,0 ,0 ,0 ,-3*L ,-L^2  ,0 ,0]
            [0 ,0 ,36  ,3*L   ,0 ,0 ,0 ,0 ,-36  ,3*L   ,0 ,0]
            [0 ,0 ,0   ,0     ,0 ,0 ,0 ,0 ,0    ,0     ,0 ,0]
            [0 ,0 ,0   ,0     ,0 ,0 ,0 ,0 ,0    ,0     ,0 ,0]
            [0 ,0 ,0   ,0     ,0 ,0 ,0 ,0 ,0    ,0     ,0 ,0]
            [0 ,0 ,0   ,0     ,0 ,0 ,0 ,0 ,0    ,0     ,0 ,0]
            [0 ,0 ,3*L ,-L^2  ,0 ,0 ,0 ,0 ,-3*L ,4*L^2 ,0 ,0])
    
        Kf = 1/(30*L)*np.array([36   ,0 ,0   ,0     ,0 ,-3*L  ,-36 ,0 ,0    ,0     ,0 ,-3*L  ]
            [0    ,0 ,0   ,0     ,0 ,0     ,0   ,0 ,0    ,0     ,0 ,0     ]
            [0    ,0 ,36  ,3*L   ,0 ,0     ,0   ,0 ,-36  ,3*L   ,0 ,0     ]
            [0    ,0 ,3*L ,4*L^2 ,0 ,0     ,0   ,0 ,-3*L ,-L^2  ,0 ,0     ]
            [0    ,0 ,0   ,0     ,0 ,0     ,0   ,0 ,0    ,0     ,0 ,0     ]
            [-3*L ,0 ,0   ,0     ,0 ,4*L^2 ,3*L ,0 ,0    ,0     ,0 ,-L^2  ]
            [-36  ,0 ,0   ,0     ,0 ,3*L   ,36  ,0 ,0    ,0     ,0 ,3*L   ]  
            [0    ,0 ,0   ,0     ,0 ,0     ,0   ,0 ,0    ,0     ,0 ,0     ] 
            [0    ,0 ,-36 ,-3*L  ,0 ,0     ,0   ,0 ,36   ,-3*L  ,0 ,0     ]    
            [0    ,0 ,3*L ,-L^2  ,0 ,0     ,0   ,0 ,-3*L ,4*L^2 ,0 ,0     ]  
            [0    ,0 ,0   ,0     ,0 ,0     ,0   ,0 ,0    ,0     ,0 ,0     ]  
            [-3*L ,0 ,0   ,0     ,0 ,-L^2  ,3*L ,0 ,0    ,0     ,0 ,4*L^2 ])
      
        Kt = -np.array([0    ,0   ,0    ,-1/L ,0  ,0    ,0    ,0  ,0    ,1/L  ,0 ,0    ],
            [0    ,0   ,0    ,0    ,0  ,0    ,0    ,0  ,0    ,0    ,0 ,0    ],
            [0    ,0   ,0    ,0    ,0  ,-1/L ,0    ,0  ,0    ,0    ,0 ,1/L  ],
            [-1/L ,0   ,0    ,0    ,0  ,-0.5 ,1/L  ,0  ,0    ,0    ,0 ,0.5  ],
            [0    ,0   ,0    ,0    ,0  ,0    ,0    ,0  ,0    ,0    ,0 ,0    ],
            [0    ,0   ,-1/L ,0.5  ,0  ,0    ,0    ,0  ,1/L  ,-0.5 ,0 ,0    ],
            [0    ,0   ,0    ,1/L  ,0  ,0    ,0    ,0  ,0    ,-1/L ,0 ,0    ],
            [0    ,0   ,0    ,0    ,0  ,0    ,0    ,0  ,0    ,0    ,0 ,0    ],
            [0    ,0   ,0    ,0    ,0  ,1/L  ,0    ,0  ,0    ,0    ,0 ,-1/L ],
            [1/L  ,0   ,0    ,0    ,0  ,-0.5 ,-1/L ,0  ,0    ,0    ,0 ,0.5  ],
            [0    ,0   ,0    ,0    ,0  ,0    ,0    ,0  ,0    ,0    ,0 ,0    ],
            [0    ,0   ,1/L  ,0.5  ,0  ,0    ,0    ,0  ,-1/L ,-0.5 ,0 ,0    ])

        K = KcEst + Kstart + Kf + Kt

        return K


    def C(self):
        """Damping matrix for an instance of a shaft element.

        Returns
        -------
        C: np.ndarray
           Damping matrix for the shaft element.

        Examples
        --------
        >>> from ross.materials import steel
        >>> # Timoshenko is the default shaft element
        >>> Timoshenko_Element = ShaftElement(0.25, 0, 0.05, steel)
        >>> # Currently internal damping for the shaft elements is not
        >>> # considered, so the matrix has only zeros.
        >>> Timoshenko_Element.C()[:4, :4]
        array([[0., 0., 0., 0.],
               [0., 0., 0., 0.],
               [0., 0., 0., 0.],
               [0., 0., 0., 0.]])
        """
        C = np.zeros((8, 8))

        return C

    def G(self):
        """Gyroscopic matrix for an instance of a shaft element.

        Returns
        -------
        G: np.ndarray
            Gyroscopic matrix for the shaft element.

        Examples
        --------
        >>> from ross.materials import steel
        >>> # Timoshenko is the default shaft element
        >>> Timoshenko_Element = ShaftElement(0.25, 0, 0.05, steel)
        >>> Timoshenko_Element.G()[:4, :4]
        array([[-0.        ,  0.01943344, -0.00022681, -0.        ],
               [-0.01943344, -0.        , -0.        , -0.00022681],
               [ 0.00022681, -0.        , -0.        ,  0.0001524 ],
               [-0.        ,  0.00022681, -0.0001524 , -0.        ]])
        """
        
        # temporary material and geometrical constants
        L = self.L
        tempG = self.material.E / (2 * (1 + self.n))
        tempS = np.pi * ((self.o_d / 2) ** 2 - (self.i_d / 2) ** 2)
        tempI = np.pi / 4 * ((self.o_d / 2) ** 4 - (self.i_d / 2) ** 4)
        tempJ = np.pi / 2 * ((self.o_d / 2) ** 4 - (self.i_d / 2) ** 4)
        tempMM = (self.i_d / 2) / (self.o_d / 2)

        # temporary variables dependent on kappa
        tempA = (
            12
            * self.material.E
            * tempI
            / (self.material.G_s * self.kappa * tempS * L ** 2)
        )

        # element level matrix declaration

        aux1 = self.material.rho * tempS * L / 420
        aux2 = self.material.rho * tempJ * L / 6

        gcor = (6/5)/(L^2*(1+tempA)^2)
        hcor = -(1/10-1/2*tempA)/(L*((1+tempA)^2))
        icor = (2/15+1/6*tempA+1/3*tempA^2)/((1+tempA)^2)
        jcor = -(1/30+1/6*tempA-1/6*tempA^2)/((1+tempA)^2)

        G = 2*self.material.rho*L*tempI*np.array([0     ,0 ,-gcor ,-hcor ,0 ,0     ,0     ,0 ,gcor  ,-hcor ,0 ,0     ]
            [gcor  ,0 ,0     ,0     ,0 ,-hcor ,-gcor ,0 ,0     ,0     ,0 ,-hcor ]
            [hcor  ,0 ,0     ,0     ,0 ,-icor ,-hcor ,0 ,0     ,0     ,0 ,-jcor ]
            [0     ,0 ,0     ,0     ,0 ,0     ,0     ,0 ,0     ,0     ,0 ,0     ]
            [0     ,0 ,hcor  ,icor  ,0 ,0     ,0     ,0 ,-hcor ,jcor  ,0 ,0     ]
            [0     ,0 ,-gcor ,-hcor ,0 ,0     ,0     ,0 ,-gcor ,hcor  ,0 ,0     ]
            [0     ,0 ,0     ,0     ,0 ,0     ,0     ,0 ,0     ,0     ,0 ,0     ]
            [gcor  ,0 ,0     ,0     ,0 ,-hcor ,gcor  ,0 ,0     ,0     ,0 ,hcor  ]
            [-hcor ,0 ,0     ,0     ,0 ,jcor  ,-hcor ,0 ,0     ,0     ,0 ,-icor ]
            [0     ,0 ,0     ,0     ,0 ,0     ,0     ,0 ,0     ,0     ,0 ,0     ]
            [0     ,0 ,0     ,0     ,0 ,0     ,0     ,0 ,0     ,0     ,0 ,0     ]
            [0     ,0 ,-hcor ,-jcor ,0 ,0     ,0     ,0 ,-hcor ,icor  ,0 ,0     ])

        return G


    def patch(self, position, check_sld, ax):
        """Shaft element patch.

        Patch that will be used to draw the shaft element.

        Parameters
        ----------
        position : float
            Position in which the patch will be drawn.
        check_sld : bool
            If True, HoverTool displays only the slenderness ratio
        ax : matplotlib axes, optional
            Axes in which the plot will be drawn.

        Returns
        -------

        """
        position_u = [position, self.i_d / 2]  # upper
        position_l = [position, -self.o_d / 2]  # lower
        width = self.L
        height = self.o_d / 2 - self.i_d / 2
        if check_sld is True and self.slenderness_ratio < 1.6:
            mpl_color = "yellow"
            legend = "Shaft - Slenderness Ratio < 1.6"
        else:
            mpl_color = self.color
            legend = "Shaft"

        #  matplotlib - plot the upper half of the shaft
        ax.add_patch(
            mpatches.Rectangle(
                position_u,
                width,
                height,
                linestyle="--",
                linewidth=0.5,
                ec="k",
                fc=mpl_color,
                alpha=0.8,
                label=legend,
            )
        )
        #  matplotlib - plot the lower half of the shaft
        ax.add_patch(
            mpatches.Rectangle(
                position_l,
                width,
                height,
                linestyle="--",
                linewidth=0.5,
                ec="k",
                fc=mpl_color,
                alpha=0.8,
            )
        )

    def bokeh_patch(self, position, check_sld, bk_ax):
        """Shaft element patch.

        Patch that will be used to draw the shaft element.

        Parameters
        ----------
        position : float
            Position in which the patch will be drawn.
        check_sld : bool
            If True, HoverTool displays only the slenderness ratio
        bk_ax : bokeh plotting axes, optional
            Axes in which the plot will be drawn.

        Returns
        -------
        hover: Bokeh HoverTool
            Bokeh HoverTool axes
        """
        if check_sld is True and self.slenderness_ratio < 1.6:
            bk_color = "yellow"
            legend = "Shaft - Slenderness Ratio < 1.6"
        else:
            bk_color = self.material.color
            legend = "Shaft"

        source_u = ColumnDataSource(
            dict(
                top=[self.o_d / 2],
                bottom=[self.i_d / 2],
                left=[position],
                right=[position + self.L],
                sld=[self.slenderness_ratio],
                elnum=[self.n],
                out_d_l=[self.o_d_l],
                out_d_r=[self.o_d_r],
                in_d_l=[self.i_d_l],
                in_d_r=[self.i_d_r],
                length=[self.L],
                mat=[self.material.name],
            )
        )

        source_l = ColumnDataSource(
            dict(
                top=[-self.o_d / 2],
                bottom=[-self.i_d / 2],
                left=[position],
                right=[position + self.L],
                sld=[self.slenderness_ratio],
                elnum=[self.n],
                out_d_l=[self.o_d_l],
                out_d_r=[self.o_d_r],
                in_d_l=[self.i_d_l],
                in_d_r=[self.i_d_r],
                length=[self.L],
                mat=[self.material.name],
            )
        )

        # bokeh plot - plot the shaft
        bk_ax.quad(
            top="top",
            bottom="bottom",
            left="left",
            right="right",
            source=source_u,
            line_color=bokeh_colors[0],
            line_width=1,
            fill_alpha=0.5,
            fill_color=bk_color,
            legend_label=legend,
            name="u_shaft",
        )
        bk_ax.quad(
            top="top",
            bottom="bottom",
            left="left",
            right="right",
            source=source_l,
            line_color=bokeh_colors[0],
            line_width=1,
            fill_alpha=0.5,
            fill_color=bk_color,
            name="l_shaft",
        )
        hover = HoverTool(names=["u_shaft", "l_shaft"])

        if check_sld:
            hover.tooltips = [
                ("Element Number :", "@elnum"),
                ("Slenderness Ratio :", "@sld"),
            ]
        else:
            hover.tooltips = [
                ("Element Number :", "@elnum"),
                ("Left Outer Diameter :", "@out_d_l"),
                ("Left Inner Diameter :", "@in_d_l"),
                ("Right Outer Diameter :", "@out_d_r"),
                ("Right Inner Diameter :", "@in_d_r"),
                ("Element Length :", "@length"),
                ("Material :", "@mat"),
            ]
        hover.mode = "mouse"

        return hover

    @classmethod
    def section(
        cls,
        L,
        ne,
        si_d,
        so_d,
        material,
        n=None,
        shear_effects=True,
        rotary_inertia=True,
        gyroscopic=True,
    ):
        """Shaft section constructor.

        This method will create a shaft section with length 'L' divided into
        'ne' elements.

        Parameters
        ----------
        i_d : float
            Inner diameter of the section.
        o_d : float
            Outer diameter of the section.
        E : float
            Young's modulus.
        G_s : float
            Shear modulus.
        material : ross.material
            Shaft material.
        n : int, optional
            Element number (coincident with it's first node).
            If not given, it will be set when the rotor is assembled
            according to the element's position in the list supplied to
            the rotor constructor.
        axial_force : float
            Axial force.
        torque : float
            Torque.
        shear_effects : bool
            Determine if shear effects are taken into account.
            Default is False.
        rotary_inertia : bool
            Determine if rotary_inertia effects are taken into account.
            Default is False.
        gyroscopic : bool
            Determine if gyroscopic effects are taken into account.
            Default is False.

        Returns
        -------
        elements: list
            List with the 'ne' shaft elements.

        Examples
        --------
        >>> # shaft material
        >>> from ross.materials import steel
        >>> # shaft inner and outer diameters
        >>> si_d = 0
        >>> so_d = 0.01585
        >>> sec = ShaftElement.section(247.65e-3, 4, 0, 15.8e-3, steel)
        >>> len(sec)
        4
        >>> sec[0].i_d
        0.0
        """

        le = L / ne

        elements = [
            cls(le, si_d, so_d, material, n, shear_effects, rotary_inertia, gyroscopic)
            for _ in range(ne)
        ]

        return elements

