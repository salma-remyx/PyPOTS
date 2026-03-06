"""
The implementation of DCdetector for the partially-observed time-series anomaly detection task.

Refer to the paper
`Yiyuan Yang, Chaoli Zhang, Tian Zhou, Qingsong Wen, and Liang Sun.
"DCdetector: Dual Attention Contrastive Representation Learning for Time Series Anomaly Detection".
In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2023.
<https://dl.acm.org/doi/10.1145/3580305.3599295>`_

"""

# Created by Yiyuan Yang <yyy1997sjz@gmail.com>
# License: BSD-3-Clause

from .model import DCdetector

__all__ = [
    "DCdetector",
]
