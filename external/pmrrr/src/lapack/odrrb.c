/* dlarrb.f -- translated by f2c (version 20061008) */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <assert.h>

/* Subroutine */ 
int odrrb(int *n, double *d__, double *lld, 
	int *ifirst, int *ilast, double *rtol1, double *rtol2, 
	 int *offset, double *w, double *wgap, double *werr, 
	double *work, int *iwork, double *pivmin, double *
	spdiam, int *twist, int *info)
{
    /* System generated locals */
    int i__1;
    double d__1, d__2;

    /* Builtin functions */
    //    double log(double);

    /* Local variables */
    int i__, k, r__, i1, ii, ip;
    double gap, mid, tmp, back, lgap, rgap, left;
    int iter, nint, prev, next;
    double cvrgd, right, width;
    extern int odneg(int *, double *, double *, double *
, double *, int *);
    int negcnt;
    double mnwdth;
    int olnint, maxitr;

/*  -- LAPACK auxiliary routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  Given the relatively robust representation(RRR) L D L^T, ODRRB */
/*  does "limited" bisection to refine the eigenvalues of L D L^T, */
/*  W( IFIRST-OFFSET ) through W( ILAST-OFFSET ), to more accuracy. Initial */
/*  guesses for these eigenvalues are input in W, the corresponding estimate */
/*  of the error in these guesses and their gaps are input in WERR */
/*  and WGAP, respectively. During bisection, intervals */
/*  [left, right] are maintained by storing their mid-points and */
/*  semi-widths in the arrays W and WERR respectively. */

/*  Arguments */
/*  ========= */

/*  N       (input) INT */
/*          The order of the matrix. */

/*  D       (input) DOUBLE PRECISION array, dimension (N) */
/*          The N diagonal elements of the diagonal matrix D. */

/*  LLD     (input) DOUBLE PRECISION array, dimension (N-1) */
/*          The (N-1) elements L(i)*L(i)*D(i). */

/*  IFIRST  (input) INT */
/*          The index of the first eigenvalue to be computed. */

/*  ILAST   (input) INT */
/*          The index of the last eigenvalue to be computed. */

/*  RTOL1   (input) DOUBLE PRECISION */
/*  RTOL2   (input) DOUBLE PRECISION */
/*          Tolerance for the convergence of the bisection intervals. */
/*          An interval [LEFT,RIGHT] has converged if */
/*          RIGHT-LEFT.LT.MAX( RTOL1*GAP, RTOL2*MAX(|LEFT|,|RIGHT|) ) */
/*          where GAP is the (estimated) distance to the nearest */
/*          eigenvalue. */

/*  OFFSET  (input) INT */
/*          Offset for the arrays W, WGAP and WERR, i.e., the IFIRST-OFFSET */
/*          through ILAST-OFFSET elements of these arrays are to be used. */

/*  W       (input/output) DOUBLE PRECISION array, dimension (N) */
/*          On input, W( IFIRST-OFFSET ) through W( ILAST-OFFSET ) are */
/*          estimates of the eigenvalues of L D L^T indexed IFIRST throug */
/*          ILAST. */
/*          On output, these estimates are refined. */

/*  WGAP    (input/output) DOUBLE PRECISION array, dimension (N-1) */
/*          On input, the (estimated) gaps between consecutive */
/*          eigenvalues of L D L^T, i.e., WGAP(I-OFFSET) is the gap between */
/*          eigenvalues I and I+1. Note that if IFIRST.EQ.ILAST */
/*          then WGAP(IFIRST-OFFSET) must be set to ZERO. */
/*          On output, these gaps are refined. */

/*  WERR    (input/output) DOUBLE PRECISION array, dimension (N) */
/*          On input, WERR( IFIRST-OFFSET ) through WERR( ILAST-OFFSET ) are */
/*          the errors in the estimates of the corresponding elements in W. */
/*          On output, these errors are refined. */

/*  WORK    (workspace) DOUBLE PRECISION array, dimension (2*N) */
/*          Workspace. */

/*  IWORK   (workspace) INT array, dimension (2*N) */
/*          Workspace. */

/*  PIVMIN  (input) DOUBLE PRECISION */
/*          The minimum pivot in the Sturm sequence. */

/*  SPDIAM  (input) DOUBLE PRECISION */
/*          The spectral diameter of the matrix. */

/*  TWIST   (input) INT */
/*          The twist index for the twisted factorization that is used */
/*          for the negcount. */
/*          TWIST = N: Compute negcount from L D L^T - LAMBDA I = L+ D+ L+^T */
/*          TWIST = 1: Compute negcount from L D L^T - LAMBDA I = U- D- U-^T */
/*          TWIST = R: Compute negcount from L D L^T - LAMBDA I = N(r) D(r) N(r) */

/*  INFO    (output) INT */
/*          Error flag. */

/*  Further Details */
/*  =============== */

/*  Based on contributions by */
/*     Beresford Parlett, University of California, Berkeley, USA */
/*     Jim Demmel, University of California, Berkeley, USA */
/*     Inderjit Dhillon, University of Texas, Austin, USA */
/*     Osni Marques, LBNL/NERSC, USA */
/*     Christof Voemel, University of California, Berkeley, USA */

/*  ===================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. External Functions .. */

/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Executable Statements .. */

    /* Parameter adjustments */
    --iwork;
    --work;
    --werr;
    --wgap;
    --w;
    --lld;
    --d__;

    /* Function Body */
    *info = 0;

    maxitr = (int) ((log(*spdiam + *pivmin) - log(*pivmin)) / log(2.)) + 
	    2;
    mnwdth = *pivmin * 2.;

    r__ = *twist;
    if (r__ < 1 || r__ > *n) {
	r__ = *n;
    }

/*     Initialize unconverged intervals in [ WORK(2*I-1), WORK(2*I) ]. */
/*     The Sturm Count, Count( WORK(2*I-1) ) is arranged to be I-1, while */
/*     Count( WORK(2*I) ) is stored in IWORK( 2*I ). The int IWORK( 2*I-1 ) */
/*     for an unconverged interval is set to the index of the next unconverged */
/*     interval, and is -1 or 0 for a converged interval. Thus a linked */
/*     list of unconverged intervals is set up. */

    i1 = *ifirst;
/*     The number of unconverged intervals */
    nint = 0;
/*     The last unconverged interval found */
    prev = 0;
    rgap = wgap[i1 - *offset];
    i__1 = *ilast;
    for (i__ = i1; i__ <= i__1; ++i__) {
	k = i__ << 1;
	ii = i__ - *offset;
	left = w[ii] - werr[ii];
	right = w[ii] + werr[ii];
	lgap = rgap;
	rgap = wgap[ii];
	gap = fmin(lgap,rgap);
/*        Make sure that [LEFT,RIGHT] contains the desired eigenvalue */
/*        Compute negcount from dstqds facto L+D+L+^T = L D L^T - LEFT */

/*        Do while( NEGCNT(LEFT).GT.I-1 ) */

	back = werr[ii];
L20:
	negcnt = odneg(n, &d__[1], &lld[1], &left, pivmin, &r__);
	if (negcnt > i__ - 1) {
	    left -= back;
	    back *= 2.;
	    goto L20;
	}

/*        Do while( NEGCNT(RIGHT).LT.I ) */
/*        Compute negcount from dstqds facto L+D+L+^T = L D L^T - RIGHT */

	back = werr[ii];
L50:
	negcnt = odneg(n, &d__[1], &lld[1], &right, pivmin, &r__);
	if (negcnt < i__) {
	    right += back;
	    back *= 2.;
	    goto L50;
	}
	width = (d__1 = left - right, fabs(d__1)) * .5;
/* Computing MAX */
	d__1 = fabs(left), d__2 = fabs(right);
	tmp = fmax(d__1,d__2);
/* Computing MAX */
	d__1 = *rtol1 * gap, d__2 = *rtol2 * tmp;
	cvrgd = fmax(d__1,d__2);
	if (width <= cvrgd || width <= mnwdth) {
/*           This interval has already converged and does not need refinement. */
/*           (Note that the gaps might change through refining the */
/*            eigenvalues, however, they can only get bigger.) */
/*           Remove it from the list. */
	    iwork[k - 1] = -1;
/*           Make sure that I1 always points to the first unconverged interval */
	    if (i__ == i1 && i__ < *ilast) {
		i1 = i__ + 1;
	    }
	    if (prev >= i1 && i__ <= *ilast) {
		iwork[(prev << 1) - 1] = i__ + 1;
	    }
	} else {
/*           unconverged interval found */
	    prev = i__;
	    ++nint;
	    iwork[k - 1] = i__ + 1;
	    iwork[k] = negcnt;
	}
	work[k - 1] = left;
	work[k] = right;
/* L75: */
    }

/*     Do while( NINT.GT.0 ), i.e. there are still unconverged intervals */
/*     and while (ITER.LT.MAXITR) */

    iter = 0;
L80:
    prev = i1 - 1;
    i__ = i1;
    olnint = nint;
    i__1 = olnint;
    for (ip = 1; ip <= i__1; ++ip) {
	k = i__ << 1;
	ii = i__ - *offset;
	rgap = wgap[ii];
	lgap = rgap;
	if (ii > 1) {
	    lgap = wgap[ii - 1];
	}
	gap = fmin(lgap,rgap);
	next = iwork[k - 1];
	left = work[k - 1];
	right = work[k];
	mid = (left + right) * .5;
/*        semiwidth of interval */
	width = right - mid;
/* Computing MAX */
	d__1 = fabs(left), d__2 = fabs(right);
	tmp = fmax(d__1,d__2);
/* Computing MAX */
	d__1 = *rtol1 * gap, d__2 = *rtol2 * tmp;
	cvrgd = fmax(d__1,d__2);
	if (width <= cvrgd || width <= mnwdth || iter == maxitr) {
/*           reduce number of unconverged intervals */
	    --nint;
/*           Mark interval as converged. */
	    iwork[k - 1] = 0;
	    if (i1 == i__) {
		i1 = next;
	    } else {
/*              Prev holds the last unconverged interval previously examined */
		if (prev >= i1) {
		    iwork[(prev << 1) - 1] = next;
		}
	    }
	    i__ = next;
	    goto L100;
	}
	prev = i__;

/*        Perform one bisection step */

	negcnt = odneg(n, &d__[1], &lld[1], &mid, pivmin, &r__);
	if (negcnt <= i__ - 1) {
	    work[k - 1] = mid;
	} else {
	    work[k] = mid;
	}
	i__ = next;
L100:
	;
    }
    ++iter;
/*     do another loop if there are still unconverged intervals */
/*     However, in the last iteration, all intervals are accepted */
/*     since this is the best we can do. */
    if (nint > 0 && iter <= maxitr) {
	goto L80;
    }


/*     At this point, all the intervals have converged */
    i__1 = *ilast;
    for (i__ = *ifirst; i__ <= i__1; ++i__) {
	k = i__ << 1;
	ii = i__ - *offset;
/*        All intervals marked by '0' have been refined. */
	if (iwork[k - 1] == 0) {
	    w[ii] = (work[k - 1] + work[k]) * .5;
	    werr[ii] = work[k] - w[ii];
	}
/* L110: */
    }

    i__1 = *ilast;
    for (i__ = *ifirst + 1; i__ <= i__1; ++i__) {
	k = i__ << 1;
	ii = i__ - *offset;
/* Computing MAX */
	d__1 = 0., d__2 = w[ii] - werr[ii] - w[ii - 1] - werr[ii - 1];
	wgap[ii - 1] = fmax(d__1,d__2);
/* L111: */
    }
    return 0;

/*     End of ODRRB */

} /* odrrb_ */
