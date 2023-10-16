#!/bin/bash

if ((PALS_LOCAL_RANKID==0)); then
  export ZE_AFFINITY_MASK=0.0
fi

if ((PALS_LOCAL_RANKID==1)); then
  export ZE_AFFINITY_MASK=0.1
fi

if ((PALS_LOCAL_RANKID==2)); then
  export ZE_AFFINITY_MASK=1.0
fi

if ((PALS_LOCAL_RANKID==3)); then
  export ZE_AFFINITY_MASK=1.1
fi

if ((PALS_LOCAL_RANKID==4)); then
  export ZE_AFFINITY_MASK=2.0
fi

if ((PALS_LOCAL_RANKID==5)); then
  export ZE_AFFINITY_MASK=2.1
fi

if ((PALS_LOCAL_RANKID==6)); then
  export ZE_AFFINITY_MASK=3.0
fi

if ((PALS_LOCAL_RANKID==7)); then
  export ZE_AFFINITY_MASK=3.1
fi

if ((PALS_LOCAL_RANKID==8)); then
  export ZE_AFFINITY_MASK=4.0
fi

if ((PALS_LOCAL_RANKID==9)); then
  export ZE_AFFINITY_MASK=4.1
fi

if ((PALS_LOCAL_RANKID==10)); then
  export ZE_AFFINITY_MASK=5.0
fi

if ((PALS_LOCAL_RANKID==11)); then
  export ZE_AFFINITY_MASK=5.1
fi

#Launch the executable:
$*


