# Makefile.in generated by automake 1.15 from Makefile.am.
# libvips/morphology/Makefile.  Generated from Makefile.in by configure.

# Copyright (C) 1994-2014 Free Software Foundation, Inc.

# This Makefile.in is free software; the Free Software Foundation
# gives unlimited permission to copy and/or distribute it,
# with or without modifications, as long as this notice is preserved.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY, to the extent permitted by law; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.




am__is_gnu_make = { \
  if test -z '$(MAKELEVEL)'; then \
    false; \
  elif test -n '$(MAKE_HOST)'; then \
    true; \
  elif test -n '$(MAKE_VERSION)' && test -n '$(CURDIR)'; then \
    true; \
  else \
    false; \
  fi; \
}
am__make_running_with_option = \
  case $${target_option-} in \
      ?) ;; \
      *) echo "am__make_running_with_option: internal error: invalid" \
              "target option '$${target_option-}' specified" >&2; \
         exit 1;; \
  esac; \
  has_opt=no; \
  sane_makeflags=$$MAKEFLAGS; \
  if $(am__is_gnu_make); then \
    sane_makeflags=$$MFLAGS; \
  else \
    case $$MAKEFLAGS in \
      *\\[\ \	]*) \
        bs=\\; \
        sane_makeflags=`printf '%s\n' "$$MAKEFLAGS" \
          | sed "s/$$bs$$bs[$$bs $$bs	]*//g"`;; \
    esac; \
  fi; \
  skip_next=no; \
  strip_trailopt () \
  { \
    flg=`printf '%s\n' "$$flg" | sed "s/$$1.*$$//"`; \
  }; \
  for flg in $$sane_makeflags; do \
    test $$skip_next = yes && { skip_next=no; continue; }; \
    case $$flg in \
      *=*|--*) continue;; \
        -*I) strip_trailopt 'I'; skip_next=yes;; \
      -*I?*) strip_trailopt 'I';; \
        -*O) strip_trailopt 'O'; skip_next=yes;; \
      -*O?*) strip_trailopt 'O';; \
        -*l) strip_trailopt 'l'; skip_next=yes;; \
      -*l?*) strip_trailopt 'l';; \
      -[dEDm]) skip_next=yes;; \
      -[JT]) skip_next=yes;; \
    esac; \
    case $$flg in \
      *$$target_option*) has_opt=yes; break;; \
    esac; \
  done; \
  test $$has_opt = yes
am__make_dryrun = (target_option=n; $(am__make_running_with_option))
am__make_keepgoing = (target_option=k; $(am__make_running_with_option))
pkgdatadir = $(datadir)/vips
pkgincludedir = $(includedir)/vips
pkglibdir = $(libdir)/vips
pkglibexecdir = $(libexecdir)/vips
am__cd = CDPATH="$${ZSH_VERSION+.}$(PATH_SEPARATOR)" && cd
install_sh_DATA = $(install_sh) -c -m 644
install_sh_PROGRAM = $(install_sh) -c
install_sh_SCRIPT = $(install_sh) -c
INSTALL_HEADER = $(INSTALL_DATA)
transform = $(program_transform_name)
NORMAL_INSTALL = :
PRE_INSTALL = :
POST_INSTALL = :
NORMAL_UNINSTALL = :
PRE_UNINSTALL = :
POST_UNINSTALL = :
build_triplet = x86_64-unknown-linux-gnu
host_triplet = x86_64-unknown-linux-gnu
subdir = libvips/morphology
ACLOCAL_M4 = $(top_srcdir)/aclocal.m4
am__aclocal_m4_deps = $(top_srcdir)/m4/gtk-doc.m4 \
	$(top_srcdir)/m4/introspection.m4 $(top_srcdir)/acinclude.m4 \
	$(top_srcdir)/configure.ac
am__configure_deps = $(am__aclocal_m4_deps) $(CONFIGURE_DEPENDENCIES) \
	$(ACLOCAL_M4)
DIST_COMMON = $(srcdir)/Makefile.am $(am__DIST_COMMON)
mkinstalldirs = $(install_sh) -d
CONFIG_HEADER = $(top_builddir)/config.h
CONFIG_CLEAN_FILES =
CONFIG_CLEAN_VPATH_FILES =
LTLIBRARIES = $(noinst_LTLIBRARIES)
libmorphology_la_LIBADD =
am_libmorphology_la_OBJECTS = morphology.lo countlines.lo rank.lo \
	hitmiss.lo morph.lo labelregions.lo
libmorphology_la_OBJECTS = $(am_libmorphology_la_OBJECTS)
AM_V_lt = $(am__v_lt_$(V))
am__v_lt_ = $(am__v_lt_$(AM_DEFAULT_VERBOSITY))
am__v_lt_0 = --silent
am__v_lt_1 = 
AM_V_P = $(am__v_P_$(V))
am__v_P_ = $(am__v_P_$(AM_DEFAULT_VERBOSITY))
am__v_P_0 = false
am__v_P_1 = :
AM_V_GEN = $(am__v_GEN_$(V))
am__v_GEN_ = $(am__v_GEN_$(AM_DEFAULT_VERBOSITY))
am__v_GEN_0 = @echo "  GEN     " $@;
am__v_GEN_1 = 
AM_V_at = $(am__v_at_$(V))
am__v_at_ = $(am__v_at_$(AM_DEFAULT_VERBOSITY))
am__v_at_0 = @
am__v_at_1 = 
DEFAULT_INCLUDES = -I. -I$(top_builddir)
depcomp = $(SHELL) $(top_srcdir)/depcomp
am__depfiles_maybe = depfiles
am__mv = mv -f
COMPILE = $(CC) $(DEFS) $(DEFAULT_INCLUDES) $(INCLUDES) $(AM_CPPFLAGS) \
	$(CPPFLAGS) $(AM_CFLAGS) $(CFLAGS)
LTCOMPILE = $(LIBTOOL) $(AM_V_lt) --tag=CC $(AM_LIBTOOLFLAGS) \
	$(LIBTOOLFLAGS) --mode=compile $(CC) $(DEFS) \
	$(DEFAULT_INCLUDES) $(INCLUDES) $(AM_CPPFLAGS) $(CPPFLAGS) \
	$(AM_CFLAGS) $(CFLAGS)
AM_V_CC = $(am__v_CC_$(V))
am__v_CC_ = $(am__v_CC_$(AM_DEFAULT_VERBOSITY))
am__v_CC_0 = @echo "  CC      " $@;
am__v_CC_1 = 
CCLD = $(CC)
LINK = $(LIBTOOL) $(AM_V_lt) --tag=CC $(AM_LIBTOOLFLAGS) \
	$(LIBTOOLFLAGS) --mode=link $(CCLD) $(AM_CFLAGS) $(CFLAGS) \
	$(AM_LDFLAGS) $(LDFLAGS) -o $@
AM_V_CCLD = $(am__v_CCLD_$(V))
am__v_CCLD_ = $(am__v_CCLD_$(AM_DEFAULT_VERBOSITY))
am__v_CCLD_0 = @echo "  CCLD    " $@;
am__v_CCLD_1 = 
SOURCES = $(libmorphology_la_SOURCES)
DIST_SOURCES = $(libmorphology_la_SOURCES)
am__can_run_installinfo = \
  case $$AM_UPDATE_INFO_DIR in \
    n|no|NO) false;; \
    *) (install-info --version) >/dev/null 2>&1;; \
  esac
am__tagged_files = $(HEADERS) $(SOURCES) $(TAGS_FILES) $(LISP)
# Read a list of newline-separated strings from the standard input,
# and print each of them once, without duplicates.  Input order is
# *not* preserved.
am__uniquify_input = $(AWK) '\
  BEGIN { nonempty = 0; } \
  { items[$$0] = 1; nonempty = 1; } \
  END { if (nonempty) { for (i in items) print i; }; } \
'
# Make sure the list of sources is unique.  This is necessary because,
# e.g., the same source file might be shared among _SOURCES variables
# for different programs/libraries.
am__define_uniq_tagged_files = \
  list='$(am__tagged_files)'; \
  unique=`for i in $$list; do \
    if test -f "$$i"; then echo $$i; else echo $(srcdir)/$$i; fi; \
  done | $(am__uniquify_input)`
ETAGS = etags
CTAGS = ctags
am__DIST_COMMON = $(srcdir)/Makefile.in $(top_srcdir)/depcomp
DISTFILES = $(DIST_COMMON) $(DIST_SOURCES) $(TEXINFOS) $(EXTRA_DIST)
ACLOCAL = ${SHELL} /srv/home/dwong/consensus-learning-paper/vips-8.2.2/missing aclocal-1.15
AMTAR = $${TAR-tar}
AM_DEFAULT_VERBOSITY = 1
AR = ar
AS = as
AUTOCONF = ${SHELL} /srv/home/dwong/consensus-learning-paper/vips-8.2.2/missing autoconf
AUTOHEADER = ${SHELL} /srv/home/dwong/consensus-learning-paper/vips-8.2.2/missing autoheader
AUTOMAKE = ${SHELL} /srv/home/dwong/consensus-learning-paper/vips-8.2.2/missing automake-1.15
AWK = gawk
CATALOGS =  en_GB.gmo de.gmo
CATOBJEXT = .gmo
CC = gcc -std=gnu99
CCDEPMODE = depmode=gcc3
CFITSIO_CFLAGS = 
CFITSIO_LIBS = 
CFLAGS = -g -O2
CPP = gcc -std=gnu99 -E
CPPFLAGS = 
CXX = g++
CXXCPP = g++ -E
CXXDEPMODE = depmode=gcc3
CXXFLAGS = -g -O2
CYGPATH_W = echo
DATADIRNAME = share
DEFS = -DHAVE_CONFIG_H
DEPDIR = .deps
DLLTOOL = dlltool
DLLWRAP = 
DSYMUTIL = 
DUMPBIN = 
ECHO_C = 
ECHO_N = -n
ECHO_T = 
EGREP = /usr/bin/grep -E
EXEEXT = 
EXIF_CFLAGS = 
EXIF_LIBS = 
EXTRA_LIBS_USED =  -ljpeg
FFTW_CFLAGS = 
FFTW_LIBS = 
FGREP = /usr/bin/grep -F
GETTEXT_PACKAGE = vips8.2
GMOFILES =  en_GB.gmo de.gmo
GMSGFMT = /srv/home/dwong/anaconda3/envs/oldpyvips/bin/msgfmt
GREP = /usr/bin/grep
GSF_CFLAGS = -I/srv/home/dwong/consensus-learning-paper/include/libgsf-1 -I/usr/include/glib-2.0 -I/usr/lib64/glib-2.0/include -I/usr/include/libxml2  
GSF_LIBS = -L/srv/home/dwong/consensus-learning-paper/lib -lgsf-1 -lgobject-2.0 -lxml2 -lglib-2.0  
GTHREAD_CFLAGS = 
GTHREAD_LIBS = 
GTKDOC_CHECK = gtkdoc-check.test
GTKDOC_CHECK_PATH = /usr/bin/gtkdoc-check
GTKDOC_DEPS_CFLAGS = -I/usr/include/glib-2.0 -I/usr/lib64/glib-2.0/include  
GTKDOC_DEPS_LIBS = -lgobject-2.0 -lglib-2.0  
GTKDOC_MKPDF = /usr/bin/gtkdoc-mkpdf
GTKDOC_REBASE = /usr/bin/gtkdoc-rebase
HTML_DIR = ${datadir}/gtk-doc/html
IMAGE_MAGICK_CFLAGS = 
IMAGE_MAGICK_LIBS = 
INSTALL = /usr/bin/install -c
INSTALL_DATA = ${INSTALL} -m 644
INSTALL_PROGRAM = ${INSTALL}
INSTALL_SCRIPT = ${INSTALL}
INSTALL_STRIP_PROGRAM = $(install_sh) -c -s
INSTOBJEXT = .mo
INTLLIBS = 
INTROSPECTION_CFLAGS = -pthread -I/usr/include/gobject-introspection-1.0 -I/usr/include/glib-2.0 -I/usr/lib64/glib-2.0/include  
INTROSPECTION_COMPILER = /usr/bin/g-ir-compiler
INTROSPECTION_GENERATE = /usr/bin/g-ir-generate
INTROSPECTION_GIRDIR = /usr/share/gir-1.0
INTROSPECTION_LIBS = -lgirepository-1.0 -lgobject-2.0 -lglib-2.0  
INTROSPECTION_MAKEFILE = /usr/share/gobject-introspection-1.0/Makefile.introspection
INTROSPECTION_SCANNER = /usr/bin/g-ir-scanner
INTROSPECTION_TYPELIBDIR = /usr/lib64/girepository-1.0
JPEG_INCLUDES = 
JPEG_LIBS = -ljpeg
LCMS_CFLAGS = 
LCMS_LIBS = 
LD = ld -m elf_x86_64
LDFLAGS = 
LIBOBJS = 
LIBRARY_AGE = 3
LIBRARY_CURRENT = 45
LIBRARY_REVISION = 2
LIBS = 
LIBTOOL = $(SHELL) $(top_builddir)/libtool
LIBWEBP_CFLAGS = 
LIBWEBP_LIBS = 
LIPO = 
LN_S = ln -s
LTLIBOBJS = 
MAGICK_CFLAGS = -fopenmp -DMAGICKCORE_HDRI_ENABLE=0 -DMAGICKCORE_QUANTUM_DEPTH=16 -I/usr/include/ImageMagick-6  
MAGICK_LIBS = -lMagickWand-6.Q16 -lMagickCore-6.Q16  
MAGICK_WAND_CFLAGS = -fopenmp -DMAGICKCORE_HDRI_ENABLE=0 -DMAGICKCORE_QUANTUM_DEPTH=16 -I/usr/include/ImageMagick-6  
MAGICK_WAND_LIBS = -lMagickWand-6.Q16 -lMagickCore-6.Q16  
MAKEINFO = ${SHELL} /srv/home/dwong/consensus-learning-paper/vips-8.2.2/missing makeinfo
MANIFEST_TOOL = :
MATIO_CFLAGS = 
MATIO_LIBS = 
MKDIR_P = /usr/bin/mkdir -p
MKINSTALLDIRS = ./mkinstalldirs
MONOTONIC_CFLAGS = -I/usr/include/glib-2.0 -I/usr/lib64/glib-2.0/include  
MONOTONIC_LIBS = -lglib-2.0  
MSGFMT = /srv/home/dwong/anaconda3/envs/oldpyvips/bin/msgfmt
MSGFMT_OPTS = -c
NM = /usr/bin/nm -B
NMEDIT = 
OBJDUMP = objdump
OBJEXT = o
OPENEXR_CFLAGS = 
OPENEXR_LIBS = 
OPENSLIDE_CFLAGS = -I/usr/include/openslide  
OPENSLIDE_LIBS = -lopenslide  
ORC_CFLAGS = 
ORC_LIBS = 
OTOOL = 
OTOOL64 = 
PACKAGE = vips
PACKAGES_USED =  glib-2.0 libxml-2.0 gmodule-2.0 gobject-2.0 libgsf-1 MagickWand openslide pangoft2 libtiff-4 libpng
PACKAGE_BUGREPORT = vipsip@jiscmail.ac.uk
PACKAGE_NAME = vips
PACKAGE_STRING = vips 8.2.2
PACKAGE_TARNAME = vips
PACKAGE_URL = 
PACKAGE_VERSION = 8.2.2
PANGOFT2_CFLAGS = -I/usr/include/pango-1.0 -I/usr/include/harfbuzz -I/usr/include/freetype2 -I/usr/include/glib-2.0 -I/usr/lib64/glib-2.0/include -I/usr/include/fribidi -I/usr/include/libpng15 -I/usr/include/uuid  
PANGOFT2_LIBS = -lpangoft2-1.0 -lpango-1.0 -lfontconfig -lgobject-2.0 -lglib-2.0 -lfreetype  
PATH_SEPARATOR = :
PKG_CONFIG = /usr/bin/pkg-config
PKG_CONFIG_LIBDIR = 
PKG_CONFIG_PATH = /srv/home/dwong/consensus-learning-paper/lib/pkgconfig/
PNG_CFLAGS = -I/usr/include/libpng15  
PNG_INCLUDES = 
PNG_LIBS = -lpng15  
POFILES =  en_GB.po de.po
POSUB = po
PO_IN_DATADIR_FALSE = 
PO_IN_DATADIR_TRUE = 
PYGOBJECT_CFLAGS = 
PYGOBJECT_LIBS = 
PYTHON = /srv/home/dwong/anaconda3/envs/oldpyvips/bin/python
PYTHON_EXEC_PREFIX = ${exec_prefix}
PYTHON_INCLUDES = -I/srv/home/dwong/anaconda3/envs/oldpyvips/include/python3.6
PYTHON_PLATFORM = linux
PYTHON_PREFIX = ${prefix}
PYTHON_VERSION = 3.6
RANLIB = ranlib
REQUIRED_CFLAGS = -pthread -I/usr/include/glib-2.0 -I/usr/lib64/glib-2.0/include -I/usr/include/libxml2  
REQUIRED_LIBS = -Wl,--export-dynamic -pthread -lgmodule-2.0 -lxml2 -lgobject-2.0 -lglib-2.0  
SED = /usr/bin/sed
SET_MAKE = 
SHELL = /bin/sh
STRIP = strip
THREADS_CFLAGS = -I/usr/include/glib-2.0 -I/usr/lib64/glib-2.0/include  
THREADS_LIBS = -lglib-2.0  
TIFF_CFLAGS =  
TIFF_INCLUDES = 
TIFF_LIBS = -ltiff  
TOP_SRCDIR = /srv/home/dwong/consensus-learning-paper/vips-8.2.2
TYPE_INIT_CFLAGS = 
TYPE_INIT_LIBS = 
USE_NLS = yes
VERSION = 8.2.2
VIPS_CFLAGS = -DG_DISABLE_ASSERT -DG_DISABLE_CHECKS -pthread -fopenmp -I/usr/lib64/glib-2.0/include -I/usr/include/uuid -I/usr/include/pango-1.0 -I/usr/include/openslide -I/usr/include/libxml2 -I/usr/include/libpng15 -I/usr/include/harfbuzz -I/usr/include/glib-2.0 -I/usr/include/fribidi -I/usr/include/freetype2 -I/usr/include/ImageMagick-6 -I/srv/home/dwong/consensus-learning-paper/include/libgsf-1 -DMAGICKCORE_QUANTUM_DEPTH=16 -DMAGICKCORE_HDRI_ENABLE=0
VIPS_ENABLE_DEPRECATED = 1
VIPS_EXEEXT = 
VIPS_INCLUDES =    
VIPS_LIBDIR = /srv/home/dwong/consensus-learning-paper/lib
VIPS_LIBS = -lMagickWand-6.Q16 -lMagickCore-6.Q16   -lpng15   -ltiff   -lz -ljpeg  -Wl,--export-dynamic -pthread -lgmodule-2.0 -lxml2 -lgobject-2.0 -lglib-2.0   -lpangoft2-1.0 -lpango-1.0 -lfontconfig -lgobject-2.0 -lglib-2.0 -lfreetype   -L/srv/home/dwong/consensus-learning-paper/lib -lgsf-1 -lgobject-2.0 -lxml2 -lglib-2.0       -lopenslide       -lm
VIPS_MAJOR_VERSION = 8
VIPS_MICRO_VERSION = 2
VIPS_MINOR_VERSION = 2
VIPS_VERSION = 8.2.2
VIPS_VERSION_STRING = 8.2.2-Tue Dec 29 12:11:30 PST 2020
XGETTEXT = /srv/home/dwong/anaconda3/envs/oldpyvips/bin/xgettext
XMKMF = 
X_CFLAGS = 
X_EXTRA_LIBS = 
X_LIBS = 
X_PRE_LIBS =  -lSM -lICE
ZIP_INCLUDES = 
ZIP_LIBS = -lz
abs_builddir = /srv/home/dwong/consensus-learning-paper/vips-8.2.2/libvips/morphology
abs_srcdir = /srv/home/dwong/consensus-learning-paper/vips-8.2.2/libvips/morphology
abs_top_builddir = /srv/home/dwong/consensus-learning-paper/vips-8.2.2
abs_top_srcdir = /srv/home/dwong/consensus-learning-paper/vips-8.2.2
ac_ct_AR = ar
ac_ct_CC = gcc
ac_ct_CXX = g++
ac_ct_DUMPBIN = 
am__include = include
am__leading_dot = .
am__quote = 
am__tar = $${TAR-tar} chof - "$$tardir"
am__untar = $${TAR-tar} xf -
bindir = ${exec_prefix}/bin
build = x86_64-unknown-linux-gnu
build_alias = 
build_cpu = x86_64
build_os = linux-gnu
build_vendor = unknown
builddir = .
datadir = ${datarootdir}
datarootdir = ${prefix}/share
docdir = ${datarootdir}/doc/${PACKAGE_TARNAME}
dvidir = ${docdir}
exec_prefix = ${prefix}
host = x86_64-unknown-linux-gnu
host_alias = 
host_cpu = x86_64
host_os = linux-gnu
host_vendor = unknown
htmldir = ${docdir}
includedir = ${prefix}/include
infodir = ${datarootdir}/info
install_sh = ${SHELL} /srv/home/dwong/consensus-learning-paper/vips-8.2.2/install-sh
libdir = ${exec_prefix}/lib
libexecdir = ${exec_prefix}/libexec
localedir = ${datarootdir}/locale
localstatedir = ${prefix}/var
mandir = ${datarootdir}/man
mkdir_p = $(MKDIR_P)
oldincludedir = /usr/include
pdfdir = ${docdir}
pkgpyexecdir = ${pyexecdir}/vips
pkgpythondir = ${pythondir}/vips
prefix = /srv/home/dwong/consensus-learning-paper
program_transform_name = s,x,x,
psdir = ${docdir}
pyexecdir = ${exec_prefix}/lib/python3.6/site-packages
pyoverridesdir = 
pythondir = ${prefix}/lib/python3.6/site-packages
sbindir = ${exec_prefix}/sbin
sharedstatedir = ${prefix}/com
srcdir = .
sysconfdir = ${prefix}/etc
target_alias = 
top_build_prefix = ../../
top_builddir = ../..
top_srcdir = ../..
vips_introspection_sources =  ./resample/affine.c ./resample/transform.c ./resample/resize.c ./resample/mapim.c ./resample/quadratic.c ./resample/interpolate.c ./resample/shrink.c ./resample/shrink2.c ./resample/resample.c ./resample/shrinkh.c ./resample/shrinkv.c ./resample/similarity.c ./arithmetic/divide.c ./arithmetic/hist_find.c ./arithmetic/deviate.c ./arithmetic/binary.c ./arithmetic/arithmetic.c ./arithmetic/unaryconst.c ./arithmetic/hough_line.c ./arithmetic/remainder.c ./arithmetic/multiply.c ./arithmetic/abs.c ./arithmetic/min.c ./arithmetic/relational.c ./arithmetic/hist_find_ndim.c ./arithmetic/hist_find_indexed.c ./arithmetic/avg.c ./arithmetic/profile.c ./arithmetic/add.c ./arithmetic/linear.c ./arithmetic/sum.c ./arithmetic/round.c ./arithmetic/subtract.c ./arithmetic/measure.c ./arithmetic/complex.c ./arithmetic/invert.c ./arithmetic/getpoint.c ./arithmetic/hough_circle.c ./arithmetic/statistic.c ./arithmetic/math2.c ./arithmetic/hough.c ./arithmetic/project.c ./arithmetic/stats.c ./arithmetic/nary.c ./arithmetic/unary.c ./arithmetic/math.c ./arithmetic/boolean.c ./arithmetic/max.c ./arithmetic/sign.c ./histogram/hist_norm.c ./histogram/hist_plot.c ./histogram/hist_ismonotonic.c ./histogram/percent.c ./histogram/histogram.c ./histogram/hist_local.c ./histogram/hist_cum.c ./histogram/stdif.c ./histogram/maplut.c ./histogram/hist_entropy.c ./histogram/hist_equal.c ./histogram/hist_unary.c ./histogram/hist_match.c ./introspect.c ./video/im_video_test.c ./video/video_dispatch.c ./convolution/im_aconv.c ./convolution/convolution.c ./convolution/im_conv.c ./convolution/gaussblur.c ./convolution/im_conv_f.c ./convolution/im_aconvsep.c ./convolution/compass.c ./convolution/fastcor.c ./convolution/conv.c ./convolution/sharpen.c ./convolution/spcor.c ./convolution/convsep.c ./convolution/correlation.c ./freqfilt/freqfilt.c ./freqfilt/phasecor.c ./freqfilt/spectrum.c ./freqfilt/freqmult.c ./freqfilt/fwfft.c ./freqfilt/invfft.c ./morphology/hitmiss.c ./morphology/morph.c ./morphology/rank.c ./morphology/countlines.c ./morphology/labelregions.c ./morphology/morphology.c ./foreign/fitssave.c ./foreign/radload.c ./foreign/rawsave.c ./foreign/ppmload.c ./foreign/fits.c ./foreign/jpegload.c ./foreign/vipssave.c ./foreign/ppm.c ./foreign/matload.c ./foreign/csvload.c ./foreign/openexr2vips.c ./foreign/webpload.c ./foreign/vips2tiff.c ./foreign/tiffsave.c ./foreign/vipspng.c ./foreign/openslideload.c ./foreign/matlab.c ./foreign/webp2vips.c ./foreign/openexrload.c ./foreign/vips2jpeg.c ./foreign/matrixsave.c ./foreign/pngsave.c ./foreign/tiff2vips.c ./foreign/vips2webp.c ./foreign/csv.c ./foreign/pngload.c ./foreign/dzsave.c ./foreign/radiance.c ./foreign/webpsave.c ./foreign/tiffload.c ./foreign/matrixload.c ./foreign/jpeg2vips.c ./foreign/csvsave.c ./foreign/magick2vips.c ./foreign/analyzeload.c ./foreign/openslide2vips.c ./foreign/fitsload.c ./foreign/foreign.c ./foreign/radsave.c ./foreign/rawload.c ./foreign/analyze2vips.c ./foreign/ppmsave.c ./foreign/magickload.c ./foreign/jpegsave.c ./foreign/vipsload.c ./colour/colourspace.c ./colour/float2rad.c ./colour/Lab2XYZ.c ./colour/LCh2UCS.c ./colour/colour.c ./colour/dE76.c ./colour/dECMC.c ./colour/scRGB2sRGB.c ./colour/Lab2LCh.c ./colour/sRGB2HSV.c ./colour/LabQ2sRGB.c ./colour/HSV2sRGB.c ./colour/sRGB2scRGB.c ./colour/scRGB2BW.c ./colour/scRGB2XYZ.c ./colour/LabQ2Lab.c ./colour/LabQ2LabS.c ./colour/Lab2LabS.c ./colour/XYZ2scRGB.c ./colour/XYZ2Lab.c ./colour/Lab2LabQ.c ./colour/rad2float.c ./colour/UCS2LCh.c ./colour/icc_transform.c ./colour/LCh2Lab.c ./colour/LabS2Lab.c ./colour/XYZ2Yxy.c ./colour/LabS2LabQ.c ./colour/dE00.c ./colour/Yxy2XYZ.c ./conversion/bandbool.c ./conversion/insert.c ./conversion/byteswap.c ./conversion/premultiply.c ./conversion/bandjoin.c ./conversion/zoom.c ./conversion/ifthenelse.c ./conversion/recomb.c ./conversion/msb.c ./conversion/rot.c ./conversion/arrayjoin.c ./conversion/replicate.c ./conversion/cache.c ./conversion/embed.c ./conversion/autorot.c ./conversion/gamma.c ./conversion/bandfold.c ./conversion/cast.c ./conversion/sequential.c ./conversion/rot45.c ./conversion/join.c ./conversion/subsample.c ./conversion/grid.c ./conversion/unpremultiply.c ./conversion/extract.c ./conversion/flip.c ./conversion/bandmean.c ./conversion/tilecache.c ./conversion/flatten.c ./conversion/conversion.c ./conversion/scale.c ./conversion/copy.c ./conversion/bandary.c ./conversion/wrap.c ./conversion/falsecolour.c ./conversion/bandunfold.c ./conversion/bandrank.c ./iofuncs/generate.c ./iofuncs/buffer.c ./iofuncs/memory.c ./iofuncs/threadpool.c ./iofuncs/mapfile.c ./iofuncs/header.c ./iofuncs/cache.c ./iofuncs/vipsmarshal.c ./iofuncs/window.c ./iofuncs/buf.c ./iofuncs/init.c ./iofuncs/sinkmemory.c ./iofuncs/system.c ./iofuncs/enumtypes.c ./iofuncs/object.c ./iofuncs/vips.c ./iofuncs/region.c ./iofuncs/sinkscreen.c ./iofuncs/sink.c ./iofuncs/type.c ./iofuncs/gate.c ./iofuncs/error.c ./iofuncs/image.c ./iofuncs/sinkdisc.c ./iofuncs/vector.c ./iofuncs/util.c ./iofuncs/operation.c ./iofuncs/rect.c ./iofuncs/base64.c ./iofuncs/semaphore.c ./create/mask_ideal_ring.c ./create/mask_fractal.c ./create/mask_ideal.c ./create/fractsurf.c ./create/mask_butterworth_band.c ./create/invertlut.c ./create/identity.c ./create/zone.c ./create/black.c ./create/buildlut.c ./create/mask_ideal_band.c ./create/tonelut.c ./create/sines.c ./create/mask_butterworth_ring.c ./create/mask_butterworth.c ./create/gaussnoise.c ./create/mask_gaussian_ring.c ./create/create.c ./create/eye.c ./create/point.c ./create/logmat.c ./create/mask_gaussian.c ./create/text.c ./create/gaussmat.c ./create/mask_gaussian_band.c ./create/xyz.c ./create/mask.c ./create/grey.c ./draw/draw_smudge.c ./draw/draw.c ./draw/draw_circle.c ./draw/draw_flood.c ./draw/drawink.c ./draw/draw_image.c ./draw/draw_mask.c ./draw/draw_line.c ./draw/draw_rect.c ./mosaicing/merge.c ./mosaicing/im_clinear.c ./mosaicing/im_remosaic.c ./mosaicing/im_avgdxdy.c ./mosaicing/im_lrmerge.c ./mosaicing/mosaicing.c ./mosaicing/im_lrmosaic.c ./mosaicing/match.c ./mosaicing/mosaic.c ./mosaicing/global_balance.c ./mosaicing/im_lrcalcon.c ./mosaicing/im_chkpair.c ./mosaicing/im_initialize.c ./mosaicing/mosaic1.c ./mosaicing/im_tbmerge.c ./mosaicing/im_tbcalcon.c ./mosaicing/im_improve.c ./mosaicing/im_tbmosaic.c include/vips/basic.h include/vips/vips.h include/vips/object.h include/vips/image.h include/vips/error.h include/vips/foreign.h include/vips/interpolate.h include/vips/header.h include/vips/operation.h include/vips/enumtypes.h include/vips/conversion.h include/vips/arithmetic.h include/vips/colour.h include/vips/convolution.h include/vips/draw.h include/vips/morphology.h include/vips/type.h include/vips/memory.h include/vips/region.h
noinst_LTLIBRARIES = libmorphology.la
libmorphology_la_SOURCES = \
	morphology.c \
	pmorphology.h \
	countlines.c \
	rank.c \
	hitmiss.c \
	morph.c \
	labelregions.c 

AM_CPPFLAGS = -I${top_srcdir}/libvips/include -DG_DISABLE_ASSERT -DG_DISABLE_CHECKS -pthread -fopenmp -I/usr/lib64/glib-2.0/include -I/usr/include/uuid -I/usr/include/pango-1.0 -I/usr/include/openslide -I/usr/include/libxml2 -I/usr/include/libpng15 -I/usr/include/harfbuzz -I/usr/include/glib-2.0 -I/usr/include/fribidi -I/usr/include/freetype2 -I/usr/include/ImageMagick-6 -I/srv/home/dwong/consensus-learning-paper/include/libgsf-1 -DMAGICKCORE_QUANTUM_DEPTH=16 -DMAGICKCORE_HDRI_ENABLE=0     
all: all-am

.SUFFIXES:
.SUFFIXES: .c .lo .o .obj
$(srcdir)/Makefile.in:  $(srcdir)/Makefile.am  $(am__configure_deps)
	@for dep in $?; do \
	  case '$(am__configure_deps)' in \
	    *$$dep*) \
	      ( cd $(top_builddir) && $(MAKE) $(AM_MAKEFLAGS) am--refresh ) \
	        && { if test -f $@; then exit 0; else break; fi; }; \
	      exit 1;; \
	  esac; \
	done; \
	echo ' cd $(top_srcdir) && $(AUTOMAKE) --foreign libvips/morphology/Makefile'; \
	$(am__cd) $(top_srcdir) && \
	  $(AUTOMAKE) --foreign libvips/morphology/Makefile
Makefile: $(srcdir)/Makefile.in $(top_builddir)/config.status
	@case '$?' in \
	  *config.status*) \
	    cd $(top_builddir) && $(MAKE) $(AM_MAKEFLAGS) am--refresh;; \
	  *) \
	    echo ' cd $(top_builddir) && $(SHELL) ./config.status $(subdir)/$@ $(am__depfiles_maybe)'; \
	    cd $(top_builddir) && $(SHELL) ./config.status $(subdir)/$@ $(am__depfiles_maybe);; \
	esac;

$(top_builddir)/config.status: $(top_srcdir)/configure $(CONFIG_STATUS_DEPENDENCIES)
	cd $(top_builddir) && $(MAKE) $(AM_MAKEFLAGS) am--refresh

$(top_srcdir)/configure:  $(am__configure_deps)
	cd $(top_builddir) && $(MAKE) $(AM_MAKEFLAGS) am--refresh
$(ACLOCAL_M4):  $(am__aclocal_m4_deps)
	cd $(top_builddir) && $(MAKE) $(AM_MAKEFLAGS) am--refresh
$(am__aclocal_m4_deps):

clean-noinstLTLIBRARIES:
	-test -z "$(noinst_LTLIBRARIES)" || rm -f $(noinst_LTLIBRARIES)
	@list='$(noinst_LTLIBRARIES)'; \
	locs=`for p in $$list; do echo $$p; done | \
	      sed 's|^[^/]*$$|.|; s|/[^/]*$$||; s|$$|/so_locations|' | \
	      sort -u`; \
	test -z "$$locs" || { \
	  echo rm -f $${locs}; \
	  rm -f $${locs}; \
	}

libmorphology.la: $(libmorphology_la_OBJECTS) $(libmorphology_la_DEPENDENCIES) $(EXTRA_libmorphology_la_DEPENDENCIES) 
	$(AM_V_CCLD)$(LINK)  $(libmorphology_la_OBJECTS) $(libmorphology_la_LIBADD) $(LIBS)

mostlyclean-compile:
	-rm -f *.$(OBJEXT)

distclean-compile:
	-rm -f *.tab.c

include ./$(DEPDIR)/countlines.Plo
include ./$(DEPDIR)/hitmiss.Plo
include ./$(DEPDIR)/labelregions.Plo
include ./$(DEPDIR)/morph.Plo
include ./$(DEPDIR)/morphology.Plo
include ./$(DEPDIR)/rank.Plo

.c.o:
	$(AM_V_CC)depbase=`echo $@ | sed 's|[^/]*$$|$(DEPDIR)/&|;s|\.o$$||'`;\
	$(COMPILE) -MT $@ -MD -MP -MF $$depbase.Tpo -c -o $@ $< &&\
	$(am__mv) $$depbase.Tpo $$depbase.Po
#	$(AM_V_CC)source='$<' object='$@' libtool=no \
#	DEPDIR=$(DEPDIR) $(CCDEPMODE) $(depcomp) \
#	$(AM_V_CC_no)$(COMPILE) -c -o $@ $<

.c.obj:
	$(AM_V_CC)depbase=`echo $@ | sed 's|[^/]*$$|$(DEPDIR)/&|;s|\.obj$$||'`;\
	$(COMPILE) -MT $@ -MD -MP -MF $$depbase.Tpo -c -o $@ `$(CYGPATH_W) '$<'` &&\
	$(am__mv) $$depbase.Tpo $$depbase.Po
#	$(AM_V_CC)source='$<' object='$@' libtool=no \
#	DEPDIR=$(DEPDIR) $(CCDEPMODE) $(depcomp) \
#	$(AM_V_CC_no)$(COMPILE) -c -o $@ `$(CYGPATH_W) '$<'`

.c.lo:
	$(AM_V_CC)depbase=`echo $@ | sed 's|[^/]*$$|$(DEPDIR)/&|;s|\.lo$$||'`;\
	$(LTCOMPILE) -MT $@ -MD -MP -MF $$depbase.Tpo -c -o $@ $< &&\
	$(am__mv) $$depbase.Tpo $$depbase.Plo
#	$(AM_V_CC)source='$<' object='$@' libtool=yes \
#	DEPDIR=$(DEPDIR) $(CCDEPMODE) $(depcomp) \
#	$(AM_V_CC_no)$(LTCOMPILE) -c -o $@ $<

mostlyclean-libtool:
	-rm -f *.lo

clean-libtool:
	-rm -rf .libs _libs

ID: $(am__tagged_files)
	$(am__define_uniq_tagged_files); mkid -fID $$unique
tags: tags-am
TAGS: tags

tags-am: $(TAGS_DEPENDENCIES) $(am__tagged_files)
	set x; \
	here=`pwd`; \
	$(am__define_uniq_tagged_files); \
	shift; \
	if test -z "$(ETAGS_ARGS)$$*$$unique"; then :; else \
	  test -n "$$unique" || unique=$$empty_fix; \
	  if test $$# -gt 0; then \
	    $(ETAGS) $(ETAGSFLAGS) $(AM_ETAGSFLAGS) $(ETAGS_ARGS) \
	      "$$@" $$unique; \
	  else \
	    $(ETAGS) $(ETAGSFLAGS) $(AM_ETAGSFLAGS) $(ETAGS_ARGS) \
	      $$unique; \
	  fi; \
	fi
ctags: ctags-am

CTAGS: ctags
ctags-am: $(TAGS_DEPENDENCIES) $(am__tagged_files)
	$(am__define_uniq_tagged_files); \
	test -z "$(CTAGS_ARGS)$$unique" \
	  || $(CTAGS) $(CTAGSFLAGS) $(AM_CTAGSFLAGS) $(CTAGS_ARGS) \
	     $$unique

GTAGS:
	here=`$(am__cd) $(top_builddir) && pwd` \
	  && $(am__cd) $(top_srcdir) \
	  && gtags -i $(GTAGS_ARGS) "$$here"
cscopelist: cscopelist-am

cscopelist-am: $(am__tagged_files)
	list='$(am__tagged_files)'; \
	case "$(srcdir)" in \
	  [\\/]* | ?:[\\/]*) sdir="$(srcdir)" ;; \
	  *) sdir=$(subdir)/$(srcdir) ;; \
	esac; \
	for i in $$list; do \
	  if test -f "$$i"; then \
	    echo "$(subdir)/$$i"; \
	  else \
	    echo "$$sdir/$$i"; \
	  fi; \
	done >> $(top_builddir)/cscope.files

distclean-tags:
	-rm -f TAGS ID GTAGS GRTAGS GSYMS GPATH tags

distdir: $(DISTFILES)
	@srcdirstrip=`echo "$(srcdir)" | sed 's/[].[^$$\\*]/\\\\&/g'`; \
	topsrcdirstrip=`echo "$(top_srcdir)" | sed 's/[].[^$$\\*]/\\\\&/g'`; \
	list='$(DISTFILES)'; \
	  dist_files=`for file in $$list; do echo $$file; done | \
	  sed -e "s|^$$srcdirstrip/||;t" \
	      -e "s|^$$topsrcdirstrip/|$(top_builddir)/|;t"`; \
	case $$dist_files in \
	  */*) $(MKDIR_P) `echo "$$dist_files" | \
			   sed '/\//!d;s|^|$(distdir)/|;s,/[^/]*$$,,' | \
			   sort -u` ;; \
	esac; \
	for file in $$dist_files; do \
	  if test -f $$file || test -d $$file; then d=.; else d=$(srcdir); fi; \
	  if test -d $$d/$$file; then \
	    dir=`echo "/$$file" | sed -e 's,/[^/]*$$,,'`; \
	    if test -d "$(distdir)/$$file"; then \
	      find "$(distdir)/$$file" -type d ! -perm -700 -exec chmod u+rwx {} \;; \
	    fi; \
	    if test -d $(srcdir)/$$file && test $$d != $(srcdir); then \
	      cp -fpR $(srcdir)/$$file "$(distdir)$$dir" || exit 1; \
	      find "$(distdir)/$$file" -type d ! -perm -700 -exec chmod u+rwx {} \;; \
	    fi; \
	    cp -fpR $$d/$$file "$(distdir)$$dir" || exit 1; \
	  else \
	    test -f "$(distdir)/$$file" \
	    || cp -p $$d/$$file "$(distdir)/$$file" \
	    || exit 1; \
	  fi; \
	done
check-am: all-am
check: check-am
all-am: Makefile $(LTLIBRARIES)
installdirs:
install: install-am
install-exec: install-exec-am
install-data: install-data-am
uninstall: uninstall-am

install-am: all-am
	@$(MAKE) $(AM_MAKEFLAGS) install-exec-am install-data-am

installcheck: installcheck-am
install-strip:
	if test -z '$(STRIP)'; then \
	  $(MAKE) $(AM_MAKEFLAGS) INSTALL_PROGRAM="$(INSTALL_STRIP_PROGRAM)" \
	    install_sh_PROGRAM="$(INSTALL_STRIP_PROGRAM)" INSTALL_STRIP_FLAG=-s \
	      install; \
	else \
	  $(MAKE) $(AM_MAKEFLAGS) INSTALL_PROGRAM="$(INSTALL_STRIP_PROGRAM)" \
	    install_sh_PROGRAM="$(INSTALL_STRIP_PROGRAM)" INSTALL_STRIP_FLAG=-s \
	    "INSTALL_PROGRAM_ENV=STRIPPROG='$(STRIP)'" install; \
	fi
mostlyclean-generic:

clean-generic:

distclean-generic:
	-test -z "$(CONFIG_CLEAN_FILES)" || rm -f $(CONFIG_CLEAN_FILES)
	-test . = "$(srcdir)" || test -z "$(CONFIG_CLEAN_VPATH_FILES)" || rm -f $(CONFIG_CLEAN_VPATH_FILES)

maintainer-clean-generic:
	@echo "This command is intended for maintainers to use"
	@echo "it deletes files that may require special tools to rebuild."
clean: clean-am

clean-am: clean-generic clean-libtool clean-noinstLTLIBRARIES \
	mostlyclean-am

distclean: distclean-am
	-rm -rf ./$(DEPDIR)
	-rm -f Makefile
distclean-am: clean-am distclean-compile distclean-generic \
	distclean-tags

dvi: dvi-am

dvi-am:

html: html-am

html-am:

info: info-am

info-am:

install-data-am:

install-dvi: install-dvi-am

install-dvi-am:

install-exec-am:

install-html: install-html-am

install-html-am:

install-info: install-info-am

install-info-am:

install-man:

install-pdf: install-pdf-am

install-pdf-am:

install-ps: install-ps-am

install-ps-am:

installcheck-am:

maintainer-clean: maintainer-clean-am
	-rm -rf ./$(DEPDIR)
	-rm -f Makefile
maintainer-clean-am: distclean-am maintainer-clean-generic

mostlyclean: mostlyclean-am

mostlyclean-am: mostlyclean-compile mostlyclean-generic \
	mostlyclean-libtool

pdf: pdf-am

pdf-am:

ps: ps-am

ps-am:

uninstall-am:

.MAKE: install-am install-strip

.PHONY: CTAGS GTAGS TAGS all all-am check check-am clean clean-generic \
	clean-libtool clean-noinstLTLIBRARIES cscopelist-am ctags \
	ctags-am distclean distclean-compile distclean-generic \
	distclean-libtool distclean-tags distdir dvi dvi-am html \
	html-am info info-am install install-am install-data \
	install-data-am install-dvi install-dvi-am install-exec \
	install-exec-am install-html install-html-am install-info \
	install-info-am install-man install-pdf install-pdf-am \
	install-ps install-ps-am install-strip installcheck \
	installcheck-am installdirs maintainer-clean \
	maintainer-clean-generic mostlyclean mostlyclean-compile \
	mostlyclean-generic mostlyclean-libtool pdf pdf-am ps ps-am \
	tags tags-am uninstall uninstall-am

.PRECIOUS: Makefile


# Tell versions [3.59,3.63) of GNU make to not export all variables.
# Otherwise a system limit (for SysV at least) may be exceeded.
.NOEXPORT:
