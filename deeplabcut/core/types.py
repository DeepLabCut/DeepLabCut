from typing import TypeAlias, Annotated

import numpy as np
from pydantic import  GetPydanticSchema, InstanceOf
from numpy.typing import NDArray


PydanticNDArray: TypeAlias = Annotated[
    NDArray,
    GetPydanticSchema(
        lambda _s, h: h(InstanceOf[np.ndarray]), lambda _s, h: h(InstanceOf[np.ndarray])
    ),
]