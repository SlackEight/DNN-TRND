from utils.wrappers import leastsquareslinefit
from numpy import arange, linspace, square, array

# compute_error functions

def sumsquared_error_regr(sequence, segment):
    """Return the sum of squared errors for a least squares line fit of one segment of a sequence"""
    x0,y0,x1,y1 = segment
    p, error = leastsquareslinefit(sequence,(x0,x1))
    return error

def sumsquared_error_int(sequence, segment):
    """Return the sum of squared errors between sequence and segment"""
    x = arange(segment[0],segment[2]+1)
    yseg = linspace(segment[1],segment[3], len(x))
    y = array(sequence[segment[0]:segment[2]+1])
    return square(y - yseg).sum()
    
# create_segment functions

def regression(sequence, seq_range):
    """Return (x0,y0,x1,y1) of a line fit to a segment of a sequence using linear regression"""
    p, error = leastsquareslinefit(sequence,seq_range)
    y0 = p[0]*seq_range[0] + p[1]
    y1 = p[0]*seq_range[1] + p[1]
    return (seq_range[0],y0,seq_range[1],y1)
    
def interpolate(sequence, seq_range):
    """Return (x0,y0,x1,y1) of a line fit to a segment using a simple interpolation"""
    return (seq_range[0], sequence[seq_range[0]], seq_range[1], sequence[seq_range[1]])
