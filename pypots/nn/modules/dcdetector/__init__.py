"""
The package including the modules of DCdetector.

Refer to the paper
`Yiyuan Yang, Chaoli Zhang, Tian Zhou, Qingsong Wen, and Liang Sun.
"DCdetector: Dual Attention Contrastive Representation Learning for Time Series Anomaly Detection".
In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2023.
<https://dl.acm.org/doi/10.1145/3580305.3599295>`_

Notes
-----
This implementation is inspired by the official one https://github.com/DAMO-DI-ML/KDD2023-DCdetector

"""

# Created by Yiyuan Yang <yyy1997sjz@gmail.com>
# License: BSD-3-Clause

from .layers import BackboneDCdetector

__all__ = [
    "BackboneDCdetector",
]
