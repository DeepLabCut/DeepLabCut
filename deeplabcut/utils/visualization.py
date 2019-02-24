"""
DeepLabCut Toolbox
https://github.com/AlexEMG/DeepLabCut
A Mathis, alexander.mathis@bethgelab.org
M Mathis, mackenzie@post.harvard.edu

"""
import os
import numpy as np
import matplotlib as mpl
import platform
from pathlib import Path

if os.environ.get('DLClight', default=False) == 'True':
    mpl.use('AGG') #anti-grain geometry engine #https://matplotlib.org/faq/usage_faq.html
    pass
elif platform.system() == 'Darwin':
    mpl.use('WXAgg')
else:
    mpl.use('TkAgg') #TkAgg
import matplotlib.pyplot as plt

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def MakeLabeledImage(DataCombined,imagenr,pcutoff,imagebasefolder,Scorers,bodyparts,colors,cfg,labels=['+','.','x'],scaling=1):
    '''Creating a labeled image with the original human labels, as well as the DeepLabCut's! '''
    from skimage import io

    alphavalue=cfg['alphavalue'] #.5
    dotsize=cfg['dotsize'] #=15

    plt.axis('off')
    im=io.imread(os.path.join(imagebasefolder,DataCombined.index[imagenr]))
    if np.ndim(im)>2: #color image!
        h,w,numcolors=np.shape(im)
    else:
        h,w=np.shape(im)
    plt.figure(frameon=False,figsize=(w*1./100*scaling,h*1./100*scaling))
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    plt.imshow(im,'gray')
    for scorerindex,loopscorer in enumerate(Scorers):
       for bpindex,bp in enumerate(bodyparts):
           if np.isfinite(DataCombined[loopscorer][bp]['y'][imagenr]+DataCombined[loopscorer][bp]['x'][imagenr]):
                y,x=int(DataCombined[loopscorer][bp]['y'][imagenr]), int(DataCombined[loopscorer][bp]['x'][imagenr])
                if 'DeepCut' in loopscorer:
                    p=DataCombined[loopscorer][bp]['likelihood'][imagenr]
                    if p>pcutoff:
                        plt.plot(x,y,labels[1],ms=dotsize,alpha=alphavalue,color=colors(int(bpindex)))
                    else:
                        plt.plot(x,y,labels[2],ms=dotsize,alpha=alphavalue,color=colors(int(bpindex)))
                else: #by exclusion this is the human labeler (I hope nobody has DeepCut in his name...)
                        plt.plot(x,y,labels[0],ms=dotsize,alpha=alphavalue,color=colors(int(bpindex)))
    plt.xlim(0,w)
    plt.ylim(0,h)
    plt.axis('off')
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    plt.gca().invert_yaxis()


def PlottingandSaveLabeledFrame(DataCombined,ind,trainIndices,cfg,colors,comparisonbodyparts,DLCscorer,foldername):
        fn=Path(cfg['project_path']+'/'+DataCombined.index[ind])
        imagename=fn.parts[-1] #fn.stem+fn.suffix
        imfoldername=fn.parts[-2] #fn.suffix
        fig=plt.figure()
        ax=fig.add_subplot(1,1,1)
        MakeLabeledImage(DataCombined,ind,cfg["pcutoff"],cfg["project_path"],[cfg["scorer"],DLCscorer],comparisonbodyparts,colors,cfg)

        if ind in trainIndices:
            full_path = os.path.join(foldername,'Training-'+imfoldername+'-'+imagename)
        else:
            full_path = os.path.join(foldername,'Test-'+imfoldername+'-'+imagename)

        # windows throws error if file path is > 260 characters, can fix with prefix. see https://docs.microsoft.com/en-us/windows/desktop/fileio/naming-a-file#maximum-path-length-limitation
        if (len(full_path) >= 260) and (os.name == 'nt'):
            full_path = '\\\\?\\'+full_path
        plt.savefig(full_path)

        plt.close("all")
