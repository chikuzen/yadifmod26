YadifMod for Avisynth 2.6.x / Avisynth+ by Chikuzen


REQUIREMENT:

      supported OS : Windows Vista sp2/ Windows7 sp1 / Windows8.1 / Windows10

      dependency : Visual C++ Redistributable for Visual Studio 2015
            (https://www.microsoft.com/en-US/download/details.aspx?id=48145)

      Avisynth 2.6.0 / Avisynth+ r1576 or greater

INFO:


      Modified version of Fizick's avisynth filter port of yadif from mplayer.  This version
   doesn't internally generate spatial predictions, but takes them from an external clip.
   It also is not an Avisynth_C plugin (just a normal one).
   This version works with all planar formats(Y8/YV12/YV411/YV16/YV24).


   Syntax =>

      yadifmod(int order, int field, int mode, clip edeint, int opt)


PARAMETERS:


   order -

      Sets the field order.

         -1 = use avisynth's internal parity value
          0 = bff
          1 = tff

      Default:  -1  (int)


   field -

      Controls which field to keep when using same rate output.  This parameter doesn't
      do anything when using double rate output.

         -1 = set equal to order
          0 = keep bottom field
          1 = keep top field

      Default:  -1  (int)


   mode -

      Controls double rate vs same rate output, and whether or not the spatial interlacing
      check is performed.

          0 = same rate, do spatial check
          1 = double rate, do spatial check
          2 = same rate, no spatial check
          3 = double rate, no spatial check

      Default:  0  (int)


   edeint -

      Clip from which to take spatial predictions.  This clip must be the same width, 
      height, and colorspace as the input clip.  If using same rate output, this clip
      should have the same number of frames as the input.  If using double rate output,
      this clip should have twice as many frames as the input.

      Default:  NULL  (clip)


   opt -

      Controls which cpu optimizations are used.

        -1 = autodetect
         0 = force c routine
         1 = force sse2 routine
         2 = force avx2 routine

      Default:  1  (int)

      Don't set this to -1 or 2 if you are using Avisynth2.6.0.
      If you use Avisynth+, not Avisynth2.6.0, there are no problems. 



NOTE:

      Don't use yadifmod_avx.dll if your machine has no AVX features.
      It was compiled with /arch:AVX .



CHANGE LIST:

   03/01/2016  yadifmod for avisynth2.6 v0.0.0

       - Initial Release by Chikuzen

   09/15/2007  v1.0

       - Initial Release by tritical


SOURCE CODE:

    https://github.com/chikuzen/yadifmod26/