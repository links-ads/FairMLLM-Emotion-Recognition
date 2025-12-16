import matplotlib.pyplot as plt
import io
import numpy as np
from PIL import Image

class FigureCollector:
    """Context manager to collect figures as images without displaying them"""
    def __init__(self):
        self.images = []
        self.titles = []
        self.original_show = None
    
    def __enter__(self):
        self.original_show = plt.show
        
        def capture_show():
            fig = plt.gcf()

            title = ""
            if fig._suptitle:
                title = fig._suptitle.get_text().split('\n')[0]
            elif fig.axes and fig.axes[0].get_title():
                title = fig.axes[0].get_title().split('\n')[0]
            self.titles.append(title)
            
            # Render figure to image with higher quality
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=200, bbox_inches='tight', facecolor='white')
            buf.seek(0)
            img = Image.open(buf)
            self.images.append(np.array(img))
            buf.close()
            plt.close(fig)
        
        plt.show = capture_show
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        plt.show = self.original_show
    
    def display_grid(self, nrows=3, ncols=2, figsize=(20, 16), title=None):
        """Display collected figures as images in a grid with improved layout"""
        n_figs = len(self.images)
        
        if n_figs == 0:
            print("No figures collected!")
            return
        
        actual_rows = min(nrows, (n_figs + ncols - 1) // ncols)
        actual_cols = min(ncols, n_figs)
        
        fig = plt.figure(figsize=figsize, facecolor='white')
        
        if title:
            fig.suptitle(title, fontsize=18, fontweight='bold', y=0.995)
        
        for idx, (img, subtitle) in enumerate(zip(self.images, self.titles)):
            if idx >= nrows * ncols:
                break
            
            ax = fig.add_subplot(actual_rows, actual_cols, idx + 1)
            ax.imshow(img)
            ax.axis('off')
            
            # Add subtitle with better formatting
            # if subtitle:
            #     ax.set_title(subtitle, fontsize=12, fontweight='bold', pad=8)
        
        for idx in range(n_figs, nrows * ncols):
            ax = fig.add_subplot(actual_rows, actual_cols, idx + 1)
            ax.axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.99] if title else [0, 0, 1, 1])
        plt.show()
        
        self.images = []
        self.titles = []