import pathlib
import tempfile

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def save_fig(f_out=None, fig=None, size_inches=(9, 6)):
    if f_out is None:
        f_out = tempfile.NamedTemporaryFile(suffix='.pdf').name
        f_out = pathlib.Path(f_out)

    if fig is None:
        fig = plt.gcf()

    with PdfPages(f_out) as pdf:
        plt.gcf().set_size_inches(*size_inches)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
