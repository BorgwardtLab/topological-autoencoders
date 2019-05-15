EXPERIMENT_CONFIGS_PATH := experiments
# Look for all config files
EXPERIMENT_CONFIGS := $(shell cd $(EXPERIMENT_CONFIGS_PATH) && find * -type f -name \*.json)

EXPERIMENT_OUTPUT_PATH := exp_runs
# Output directories for the individual experiments (remove .json)
RUN_EXPERIMENTS := $(foreach exp_config,$(EXPERIMENT_CONFIGS),$(EXPERIMENT_OUTPUT_PATH)/$(subst .json,,$(exp_config)))
#$(info $$RUN_EXPERIMENTS is [${RUN_EXPERIMENTS}])

FILTER := 
FILTERED_EXPERIMENTS := $(shell echo $(RUN_EXPERIMENTS) | tr ' ' '\n' | grep $(FILTER))

.PHONY: help all filtered


help:
	@# Got this from https://gist.github.com/rcmachado/af3db315e31383502660#gistcomment-1585632
	$(info Available targets)
	@awk '/^[a-zA-Z\-\_0-9]+:/ {                    \
	  nb = sub( /^## /, "", helpMsg );              \
	  if(nb == 0) {                                 \
	    helpMsg = $$0;                              \
	    nb = sub( /^[^:]*:.* ## /, "", helpMsg );   \
	  }                                             \
	  if (nb)                                       \
	    print  $$1 helpMsg;                         \
	}                                               \
	{ helpMsg = $$0 }'                              \
	$(MAKEFILE_LIST) | column -ts:

## Run all experiments defined in $(EXPERIMENT_CONFIGS_PATH)
all: $(RUN_EXPERIMENTS)

filtered: $(FILTERED_EXPERIMENTS)

$(EXPERIMENT_OUTPUT_PATH)/%: $(EXPERIMENT_CONFIGS_PATH)/%.json
	@# Split of the highest level directory besides $(EXPERIMENT_CONFIGS_PATH) and interpret it 
	@# as the script we want to run
	-@sacred_experiment=$(word 2,$(subst /, ,$<)); \
	if [ -e exp/$${sacred_experiment}.py ]; then \
	echo python -m exp.$${sacred_experiment} -e -F $@ with $< $(SACRED_OVERRIDES); \
	mkdir -p $@; \
	python -m exp.$${sacred_experiment} -F $@ with $< $(SACRED_OVERRIDES) && mv $@/1/* $@ && rm -r $@/1; \
	fi

