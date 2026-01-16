# start with working directory: scenario_gender
# dataset should be prepared before this script

# There are 3 actions possible actions if a `lightning_logs` 
# or `conf/embeddings_validation.work` folder already exists:
# - delete: delete the existing folder
# - raise: raise an error and stop the script
# - keep: keep the existing folder


# Usage:
#   sh bin/run-all-scenarios --action-on-exist delete
#   sh bin/run-all-scenarios --action-on-exist keep
#   sh bin/run-all-scenarios --action-on-exist raise






# Function to check and handle folders
check_and_handle_folder() {
    local folder_path=$1
    local action_on_exist=$2

    if [ -d "$folder_path" ]; then
        case $action_on_exist in
            delete)
                echo "Deleted existing folder: $folder_path"
                rm -rf "$folder_path"
                ;;
            raise)
                echo "Folder already exists: $folder_path. Delete it and rerun the script with option \`--action-on-exist keep\` or \`--action-on-exist delete\`."
                exit 1
                ;;
            keep)
                echo "Keeping existing folder: $folder_path"
                ;;
            *)
                echo "Unknown action: $action_on_exist"
                exit 1
                ;;
        esac
    fi
}

# Default value for action_on_exist
action_on_exist="raise"

# Parse command-line arguments
while [ "$#" -gt 0 ]; do
    case $1 in
        --action-on-exist)
            action_on_exist="$2"
            shift
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
    shift
done





# Check and handle folders
check_and_handle_folder "lightning_logs" "$action_on_exist"
check_and_handle_folder "conf/embeddings_validation.work" "$action_on_exist"

# Start script execution
echo "==== Folds split"
python -m embeddings_validation --config-dir conf --config-name embeddings_validation_baselines_supervised +workers=1 +total_cpu_count=4 +split_only=True


echo "==== Device cuda:${CUDA_VISIBLE_DEVICES} will be used"
echo ""
echo "==== Start"

# echo "==== Scenario: COLEs with GNN 0_1 \n\n"
# sh bin/scenario_coles_gnn__g_0_1__no_orig_emb__graph_sage.sh

# echo "==== Scenario: COLEs with GNN 0_5 \n\n"
# sh bin/scenario_coles_gnn__g_0_5__no_orig_emb__graph_sage.sh

# echo "==== Scenario: COLEs with GNN 0_9 \n\n"
# sh bin/scenario_coles_gnn__g_0_9__no_orig_emb__graph_sage.sh


# echo "==== Scenario: COLEs with GNN 0_1 \n\n"
# sh bin/scenario_coles_gnn__g_0_1__has_orig_emb__graph_sage.sh

# echo "==== Scenario: COLEs with GNN 0_5 \n\n"
# sh bin/scenario_coles_gnn__g_0_5__has_orig_emb__graph_sage.sh

# echo "==== Scenario: COLEs with GNN 0_9 \n\n"
# sh bin/scenario_coles_gnn__g_0_9__has_orig_emb__graph_sage.sh





echo "==== Scenario: COLEs with GNN 0_1 \n\n"
sh bin/scenario_coles_gnn__g_0_1__no_orig_emb__graph_sage.sh

echo "==== Scenario: COLEs with GNN 0_5 \n\n"
sh bin/scenario_coles_gnn__g_0_5__no_orig_emb__graph_sage.sh

echo "==== Scenario: COLEs with GNN 0_9 \n\n"
sh bin/scenario_coles_gnn__g_0_9__no_orig_emb__graph_sage.sh


echo "==== Scenario: COLEs with GNN 0_1 \n\n"
sh bin/scenario_coles_gnn__g_0_1__has_orig_emb__graph_sage.sh

echo "==== Scenario: COLEs with GNN 0_5 \n\n"
sh bin/scenario_coles_gnn__g_0_5__has_orig_emb__graph_sage.sh

echo "==== Scenario: COLEs with GNN 0_9 \n\n"
sh bin/scenario_coles_gnn__g_0_9__has_orig_emb__graph_sage.sh

# Compare
rm results/scenario_gender_2024_research.txt -f
# rm -r conf/embeddings_validation.work/
python -m embeddings_validation --config-dir conf --config-name embeddings_validation__2024_research +workers=1 +total_cpu_count=4
