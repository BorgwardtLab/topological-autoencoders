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

# Do sharding in case we are using multiple GPUs
N_SHARD := 1
SHARD_INDEX := 0
shard = $(shell python -m scripts.shard $(1) --nshard $(N_SHARD) --index $(SHARD_INDEX))


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
all: $(call shard, $(RUN_EXPERIMENTS))

filtered: $(call shard, $(FILTERED_EXPERIMENTS))

$(EXPERIMENT_OUTPUT_PATH)/%: $(EXPERIMENT_CONFIGS_PATH)/%.json
	@# Split of the highest level directory besides $(EXPERIMENT_CONFIGS_PATH) and interpret it 
	@# as the script we want to run
	@# export CUDA_VISIBLE_DEVICES=$$(python -c 'import GPUtil; print(GPUtil.getAvailable("random")[0])');
	-@sacred_experiment=$(word 2,$(subst /, ,$<)); \
	if [ ! -e $@ ]; then \
	if [ -e exp/$${sacred_experiment}.py ]; then \
	echo python -m exp.$${sacred_experiment} -e -F $@ with $< $(SACRED_OVERRIDES); \
	mkdir -p $@; \
	python -m exp.$${sacred_experiment} -F $@ with $< $(SACRED_OVERRIDES) && mv $@/1/* $@ && rm -r $@/1; \
	fi; \
	else \
	echo $@ already exists! Skipping... ;\
	fi

