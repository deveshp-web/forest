''' Workflow for processing raw Beiwe tracking survey metadata with powrep.

'''
###############################################################################
# 0. Set up paths
###############################################################################
raw_dir  = 'path/to/directory/containing/raw/Beiwe/user/data'
proc_dir = 'path/to/directory/for/powrep/output'

###############################################################################
# 1. Create a BeiweProject
###############################################################################
from beiwetools.manage import BeiweProject
p = BeiweProject.create(raw_dir)

###############################################################################
# 2. Summarize power state data
# - This is mainly for preliminary data exploration and troubleshooting.
###############################################################################
from survrep import pack_summary_kwargs, Summary
# pack arguments for Summary.do():
user_ids = p.ids # summarize metadata for all available Beiwe IDs
dir_names = p.surveys['survey_timings'] # summarize metadata for all available
                                        # survey timings directories-- highly 
                                        # recommended, because many directories 
                                        # may contain metadata for other surveys
project = p
track_time = True # isolate output in a timestamped directory
kwargs = pack_summary_kwargs(user_ids, proc_dir, dir_names, 
                             project, track_time)    
# run the summary:
Summary.do(**kwargs)

# Now review proc_dir/survrep/summary/<timestamp>/log/log.csv:
#   - Check for log records with "WARNING" in the levelname column.
#     The most common warning message may be "Unrecognized survey."
#       - This means that an event was found from a survey that doesn't match
#         the directory name in which it was logged.
#       - This isn't too much of a concern, as this package's Extract module
#         ignores directory names when it organizes event records.

# Summaries for each survey identifer in dir_names are found in 
# proc_dir/survrep/summary/<timestamp>/records/records.csv:
#   - Check for survey identifiers and Beiwe identifiers with unknown headers
#     or unknown events.
#       - If necessary, review log.csv for details regarding these issues.
#       - New events should be documented and added to powrep/events.json.
#       - Update powrep/header.py with new header information if necessary.

###############################################################################
# 3. Check compatibility with a configuration file
#   - This is an optional step.
###############################################################################







###############################################################################
# 4. Extract event variables
#   - This module organizes power state events into variables.
#   - For example, iOS Unlocked/Locked events are converted to a boolean
#     variable called "protected_data_available".
#   - Each event variable can be extracted to a CSV.
###############################################################################
from powrep import var_names, pack_extract_kwargs, Extract
# review available variables:
for opsys in ['iOS', 'Android']:
    print(var_names[opsys])
# pack arguments for Extract.do():
kwargs = pack_extract_kwargs(user_ids = p.ids, 
                             proc_dir = proc_dir, 
                             project = p, 
                             get_variables = [], # This will extract all 
                                                 # variables.
                             track_time = True)
# run the variable extraction:
Extract.do(**kwargs)

# Now review proc_dir/extract/log/log.csv:
#   - Check for log records with "WARNING" in the levelname column.
#   - The main warning to watch for is "Unable to extract."

# Extracted variables are found in proc_dir/extract/data/<user_id>:
#   - Observations of each variable are in <variable name>.csv.
#   - CSV headers are 'timestamp' and 'value'; iOS files have an additional
#     'battery_level' column.