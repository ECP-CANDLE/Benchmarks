#!/bin/bash

if ((PALS_LOCAL_RANKID==0)); then
  export ZE_AFFINITY_MASK=0.0
fi

if ((PALS_LOCAL_RANKID==1)); then
  export ZE_AFFINITY_MASK=0.0
fi

if ((PALS_LOCAL_RANKID==2)); then
  export ZE_AFFINITY_MASK=0.0
fi

if ((PALS_LOCAL_RANKID==3)); then
  export ZE_AFFINITY_MASK=0.0
fi

if ((PALS_LOCAL_RANKID==4)); then
  export ZE_AFFINITY_MASK=0.1
fi

if ((PALS_LOCAL_RANKID==5)); then
  export ZE_AFFINITY_MASK=0.1
fi

if ((PALS_LOCAL_RANKID==6)); then
  export ZE_AFFINITY_MASK=0.1
fi

if ((PALS_LOCAL_RANKID==7)); then
  export ZE_AFFINITY_MASK=0.1
fi

if ((PALS_LOCAL_RANKID==8)); then
  export ZE_AFFINITY_MASK=1.0
fi

if ((PALS_LOCAL_RANKID==9)); then
  export ZE_AFFINITY_MASK=1.0
fi

if ((PALS_LOCAL_RANKID==10)); then
  export ZE_AFFINITY_MASK=1.0
fi

if ((PALS_LOCAL_RANKID==11)); then
  export ZE_AFFINITY_MASK=1.0
fi

if ((PALS_LOCAL_RANKID==12)); then
  export ZE_AFFINITY_MASK=1.1
fi

if ((PALS_LOCAL_RANKID==13)); then
  export ZE_AFFINITY_MASK=1.1
fi

if ((PALS_LOCAL_RANKID==14)); then
  export ZE_AFFINITY_MASK=1.1
fi

if ((PALS_LOCAL_RANKID==15)); then
  export ZE_AFFINITY_MASK=1.1
fi

if ((PALS_LOCAL_RANKID==16)); then
  export ZE_AFFINITY_MASK=2.0
fi

if ((PALS_LOCAL_RANKID==17)); then
  export ZE_AFFINITY_MASK=2.0
fi

if ((PALS_LOCAL_RANKID==18)); then
  export ZE_AFFINITY_MASK=2.0
fi

if ((PALS_LOCAL_RANKID==19)); then
  export ZE_AFFINITY_MASK=2.0
fi

if ((PALS_LOCAL_RANKID==20)); then
  export ZE_AFFINITY_MASK=2.1
fi

if ((PALS_LOCAL_RANKID==21)); then
  export ZE_AFFINITY_MASK=2.1
fi

if ((PALS_LOCAL_RANKID==22)); then
  export ZE_AFFINITY_MASK=2.1
fi

if ((PALS_LOCAL_RANKID==23)); then
  export ZE_AFFINITY_MASK=2.1
fi

if ((PALS_LOCAL_RANKID==24)); then
  export ZE_AFFINITY_MASK=3.0
fi

if ((PALS_LOCAL_RANKID==25)); then
  export ZE_AFFINITY_MASK=3.0
fi

if ((PALS_LOCAL_RANKID==26)); then
  export ZE_AFFINITY_MASK=3.0
fi

if ((PALS_LOCAL_RANKID==27)); then
  export ZE_AFFINITY_MASK=3.0
fi

if ((PALS_LOCAL_RANKID==28)); then
  export ZE_AFFINITY_MASK=3.1
fi

if ((PALS_LOCAL_RANKID==29)); then
  export ZE_AFFINITY_MASK=3.1
fi

if ((PALS_LOCAL_RANKID==30)); then
  export ZE_AFFINITY_MASK=3.1
fi

if ((PALS_LOCAL_RANKID==31)); then
  export ZE_AFFINITY_MASK=3.1
fi

if ((PALS_LOCAL_RANKID==32)); then
  export ZE_AFFINITY_MASK=4.0
fi

if ((PALS_LOCAL_RANKID==33)); then
  export ZE_AFFINITY_MASK=4.0
fi

if ((PALS_LOCAL_RANKID==34)); then
  export ZE_AFFINITY_MASK=4.0
fi

if ((PALS_LOCAL_RANKID==35)); then
  export ZE_AFFINITY_MASK=4.0
fi

if ((PALS_LOCAL_RANKID==36)); then
  export ZE_AFFINITY_MASK=4.1
fi

if ((PALS_LOCAL_RANKID==37)); then
  export ZE_AFFINITY_MASK=4.1
fi

if ((PALS_LOCAL_RANKID==38)); then
  export ZE_AFFINITY_MASK=4.1
fi

if ((PALS_LOCAL_RANKID==39)); then
  export ZE_AFFINITY_MASK=4.1
fi

if ((PALS_LOCAL_RANKID==40)); then
  export ZE_AFFINITY_MASK=5.0
fi

if ((PALS_LOCAL_RANKID==41)); then
  export ZE_AFFINITY_MASK=5.0
fi

if ((PALS_LOCAL_RANKID==42)); then
  export ZE_AFFINITY_MASK=5.0
fi

if ((PALS_LOCAL_RANKID==43)); then
  export ZE_AFFINITY_MASK=5.0
fi

if ((PALS_LOCAL_RANKID==44)); then
  export ZE_AFFINITY_MASK=5.1
fi

if ((PALS_LOCAL_RANKID==45)); then
  export ZE_AFFINITY_MASK=5.1
fi

if ((PALS_LOCAL_RANKID==46)); then
  export ZE_AFFINITY_MASK=5.1
fi

if ((PALS_LOCAL_RANKID==47)); then
  export ZE_AFFINITY_MASK=5.1
fi


#Launch the executable:
$*


