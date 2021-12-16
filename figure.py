"""
Contains method definitions for generating figures
"""
from core import * 

def getPositiveAnnotationDistribution(exclude_zero=True):
    """
    get graphs for the counts of images that were given a positive vote by exactly n annotators
    """
    USERS = ['NP1', 'NP2', 'NP3', 'NP4', 'NP5']
    amyloid_types = ["cored", "diffuse", "CAA"]
    ##get df of image, and number of positive annotations (max is 5) it got for each class 
    user_img_label_mapp = {u: {} for u in USERS} #key: user, key: image name, value: label tuple
    for user in USERS:
        df = pd.read_csv("csvs/phase1/binary_labels/phase1_labels_{}.csv".format(user))        
        for index, row in df.iterrows():
            user_img_label_mapp[row["username"]][row["imagename"]] = (row["cored"], row["diffuse"], row["CAA"], row["negative"],row["flag"],row["notsure"])
    phase1_images = set(df["imagename"])
    new_entries = []
    for img in phase1_images:
        cored_count, diffuse_count, CAA_count, negative_count, flag_count, not_sure_count = 0,0,0,0,0,0
        for user in USERS:
            cored_count += user_img_label_mapp[user][img][0]
            diffuse_count += user_img_label_mapp[user][img][1]
            CAA_count += user_img_label_mapp[user][img][2]
            negative_count += user_img_label_mapp[user][img][3]
            flag_count += user_img_label_mapp[user][img][4]
            not_sure_count += user_img_label_mapp[user][img][5]
        source = img[0:img.find("/")]
        new_entries.append((img, cored_count, diffuse_count, CAA_count, negative_count, flag_count, not_sure_count, source))
    phase1exact_counts = pd.DataFrame(new_entries, columns =['imagename', 'cored', 'diffuse', 'CAA', "negative", "flag", "notsure", "source"])
    ##convert to dictionary of: key: amyloid class, key: exact n labeled positive (0->5), value: count 
    exact_count_dict = {am_type: {i: 0 for i in range(0, 6)} for am_type in amyloid_types} 
    for index, row in phase1exact_counts.iterrows():
        for am_type in amyloid_types:
            count = row[am_type]
            exact_count_dict[am_type][count] += 1
    ##dictionary to hold total sums for each class, so that we can plot as percentage 
    summations = {am_type: 0 for am_type in amyloid_types}
    ##make bar graphs 
    all_class_percentages_list = [] #list of lists: [[x1, x2, ... x5 for cored], [diffuse], [CAA]] of percentages for each x in 1...5
    for am_type in amyloid_types:
        if exclude_zero:
            del exact_count_dict[am_type][0]
        x = list(exact_count_dict[am_type].keys()) #0->5 or 1->5
        y = [exact_count_dict[am_type][i] for i in list(exact_count_dict[am_type].keys())]
        summation = sum(y)
        summations[am_type] = summation
        percentages = [el / float(summation) for el in y]
        all_class_percentages_list.append(percentages)
    ##make summary bar graph of all 3 classes in one graph
    plt.cla()
    plt.clf()
    fig, ax = plt.subplots()
    width = .25
    colors = ["maroon", "bisque", "darkorange"]
    x = np.array(x)
    xlabels = ['null'] + list(x)
    classes = ["Cored", "Diffuse", "CAA"]
    for i in range(0, 3):
        rect = ax.bar(x + width * i, all_class_percentages_list[i], width, color=colors[i], label=classes[i] + ", 100% = {} images".format(summations[amyloid_types[i]]))
        autolabel(rect, ax, percentage=True, fontsize=8)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.85])
    ax.legend(loc='upper right', fontsize=10, bbox_to_anchor=(1, 1.35))
    plt.gcf().subplots_adjust(bottom=0.14, top=.76)
    ax.set_ylim((0,1))
    ax.set_xticklabels(xlabels,fontsize=12)
    ax.set_xlabel("Exact Number of Annotators Who Labeled as Positive\n(e)", fontsize=12)
    ax.set_ylabel("Percentage", fontsize=12)
    y_vals = ax.get_yticks()
    ax.set_yticklabels(['{:,.0%}'.format(y) for y in y_vals], fontsize=12)
    plt.title("Distribution of Positive Annotations", fontsize=14)   
    plt.savefig("figures/positive_distribution_exclude_zero_{}_all_classes.png".format(exclude_zero), dpi=300)


def getUsersClassCounts():
    """
    Plots a graph of average class prevalence by annotations 
    """
    USERS = ['NP1', 'NP2', 'NP3', 'NP4', 'NP5', 'UG1', 'UG2']
    am_types = {"cored":0, "diffuse":1, "CAA":2}
    class_counts = {u: [0,0,0] for u in USERS} #key: user, value: list of class counts 
    for user in USERS:
        df = pd.read_csv("csvs/phase1/binary_labels/phase1_labels_{}.csv".format(user))
        for index, row in df.iterrows():
            for am in ["cored", "diffuse", "CAA"]:
                class_counts[user][am_types[am]] += row[am]
    all_user_percentages = [] 
    for user in USERS:
        y = class_counts[user]
        summation = sum(y)
        percentages = [el / float(summation) for el in y]
        all_user_percentages.append(percentages)
    ##make summary graph
    fig, ax = plt.subplots()
    width = .27
    x = list(range(3))
    xlabels = ["null", "Cored", "Diffuse", "CAA"]    
    means = [np.mean([x[i] for x in all_user_percentages]) for i in range(0, 3)]
    stds = [np.std([x[i] for x in all_user_percentages]) for i in range(0, 3)]
    ax.bar(x, means, width, yerr=stds, capsize=3, color=("maroon", "bisque", "darkorange"))
    for i,j in zip(x, means):
        ax.annotate("{:.0%}".format(j), xy=(i - .03, j +.03),fontsize=12)
    ax.set_xticklabels(xlabels, fontsize=12)
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.set_ylabel("Percentage", fontsize=12)
    ax.set_ylim((0,1))  
    plt.title("Average Class Prevalence by Annotations", fontsize=14, y=1.02)
    y_vals = ax.get_yticks()
    ax.set_yticklabels(['{:,.0%}'.format(y) for y in y_vals], fontsize=12)
    plt.savefig("figures/class_counts_all_users.jpg", dpi=300)

def vennDiagramPhase1():
    """
    Plots venn diagram of agreement by class
    """
    USERS = ['NP1', 'NP2', 'NP3', 'NP4', 'NP5']
    #dictionaries mapping user to list of images that has this class labeled as (+)
    cored = {} 
    caa = {}
    diffuse = {}
    negative = {}
    for user in USERS:
        df = pd.read_csv("csvs/phase1/binary_labels/phase1_labels_{}.csv".format(user))
        cored[user] = set(list(df[df["cored"]==1]["imagename"]))
        caa[user] = set(list(df[df["CAA"]==1]["imagename"]))
        diffuse[user] = set(list(df[df["diffuse"]==1]["imagename"]))
        negative[user] = set(list(df[df["negative"]==1]["imagename"]))
    dictionary = {"cored": cored, "diffuse": diffuse, "caa": caa}
    for amy_class in ["cored","diffuse", "caa"]:
        labels = venn.get_labels([list(dictionary[amy_class][user]) for user in USERS], fill=["number"])
        venn_summation = sum([float(labels[key]) for key in labels.keys()])
        fig, ax = venn.venn5(labels, names=["NP"+str(i) for i in range(1,6)])
        fig.savefig("figures/venn_diagram_{}.png".format(amy_class), dpi=300)
        plt.cla()
        plt.clf()
        plt.close()

def plotInterraterAgreement(exclude_amateurs=False, phase="phase1", test_set_only=False):
    """
    Plots a grid of interrater agreement
    Requires "pickles/{PHASE}_test_set_kappa_dict.pk"
    """
    am_types = ["cored", "diffuse", "CAA"]
    if exclude_amateurs:
        USERS = ['NP1', 'NP2', 'NP3', 'NP4', 'NP5']
    else:
        USERS = ['UG1', 'UG2', 'NP1', 'NP2', 'NP3', 'NP4', 'NP5']
    kappa_dict = pickle.load(open("pickles/{}_kappa_dict_exclude_novices_{}_test_set_only_{}.pk".format(phase, exclude_amateurs, test_set_only), "rb"))
    accuracy_dict = pickle.load(open("pickles/{}_accuracy_dict_exclude_novices_{}_test_set_only_{}.pk".format(phase, exclude_amateurs, test_set_only), "rb"))
    for score_type in ["kappa", "accuracy"]:
        if score_type == "kappa":
            dictionary = kappa_dict
        if score_type == "accuracy":
            dictionary = accuracy_dict
        for am_type in am_types:
            grid = [] #2D grid of x: csv dataset, y: model, value: AUPRC score
            for user1 in USERS:
                results = []
                for user2 in USERS:
                    try:
                        results.append(float(str(dictionary[(user1,user2)][am_type])[0:5]))
                    except:
                        results.append(float(str(dictionary[(user2,user1)][am_type])[0:5]))
                grid.append(results)
            grid = np.array(grid)
            elements = []
            for row in grid:
                for element in row:
                    if element != 1:
                        elements.append(element)
            class_average = np.mean(elements)
            class_std = np.std(elements)
            fig, ax = plt.subplots()
            im = ax.imshow(grid,vmin=.15, vmax=1)
            ax.set_xticks(np.arange(len(USERS)))
            ax.set_yticks(np.arange(len(USERS)))
            plt_labels = USERS
            ax.set_xticklabels(plt_labels,fontsize=11)
            ax.set_yticklabels(plt_labels,fontsize=11)
            # Loop over data dimensions and create text annotations.
            for i in range(len(USERS)):
                for j in range(len(USERS)):
                    text = ax.text(j, i, str(grid[i, j])[0:4], ha="center", va="center", color="black", fontsize=11)
                    # text.set_path_effects([path_effects.Stroke(linewidth=.8, foreground='black'), path_effects.Normal()])
            cbar = ax.figure.colorbar(im, ax=ax)
            cbar.ax.tick_params(labelsize=11)
            fig.tight_layout()
            if am_type in ["cored", "diffuse"]:
                am_type = am_type.capitalize()
            if test_set_only:
                ax.set_title("{} Test Set Only {} Statistics, Class: {}".format(phase.capitalize(), score_type.capitalize(), am_type), fontsize=12)
                plt.savefig("figures/interrater_{}_test_set_{}_stats_{}.png".format(phase, score_type, am_type), dpi=300)
            else:
                ax.set_title("{} Plaques ({} ± {})".format(am_type, class_average.round(decimals=2), class_std.round(decimals=2)), fontsize=12)
                plt.savefig("figures/interrater_{}_{}_stats_{}.png".format(phase, score_type, am_type), dpi=300)
            
def getAverageIndividualOrConsensusMetrics():
    """
    for each of the 3 amyloid classes, will plot average AUROC and average AUPRC over inidividual models, consensus models, and random baseline 
    excludes amateurs
    """
    metrics = ["AUPRC", "AUROC"]
    classes = [0, 1, 2]
    AUPRCs = []
    AUROCS = []
    evals = ["all", "individual", "consensus"]
    diagonals_mapp = {e: {metric: [] for metric in metrics} for e in evals} #key: eval, key: metric, value: [(avg, std) for each amyloid class]
    for ev in evals:
        for metric in metrics:
            for c in classes:
                mapp = pickle.load(open("pickles/mapp_{} mapval_type{}_class_{}_random_ensemble_False_multiple_randoms_False.p".format(metric, "test_set", c), "rb"))
                ##exclude things like ensembles, random models and truths, and amateurs
                mapp = {k: mapp[k] for k in mapp.keys() if "ensemble" not in k[0] and "random" not in k[0] and "random" not in k[1]
                    and "UG1" not in k[0] and "UG1" not in k[1]
                    and "UG2" not in k[0] and "UG2" not in k[1]}
                if ev == "individual":
                    mapp = {k: mapp[k] for k in mapp.keys() if "thresholding" not in getUser(k[0]) and "thresholding" not in getUser(k[1])}
                if ev == "consensus":
                    mapp = {k: mapp[k] for k in mapp.keys() if "thresholding" in getUser(k[0]) and "thresholding" in getUser(k[1])}
                diagonals_only_mapp = {k: mapp[k] for k in mapp.keys() if getUser(k[0]) == getUser(k[1])}
                average = np.mean([diagonals_only_mapp[k][0] for k in diagonals_only_mapp])
                std =   np.std([diagonals_only_mapp[k][0] for k in diagonals_only_mapp])
                diagonals_mapp[ev][metric].append((average, std))           
    ##make stacked bar graph of individuals and consensus
    cons_AUPRCs = [i[0] for i in diagonals_mapp["consensus"]["AUPRC"]]
    cons_AUPRC_errs = [i[1] for i in diagonals_mapp["consensus"]["AUPRC"]]
    cons_AUROCs = [i[0] for i in diagonals_mapp["consensus"]["AUROC"]]
    cons_AUROC_errs = [i[1] for i in diagonals_mapp["consensus"]["AUROC"]]
    individ_AUPRCs = [i[0] for i in diagonals_mapp["individual"]["AUPRC"]]
    individ_AUPRC_errs = [i[1] for i in diagonals_mapp["individual"]["AUPRC"]]
    individ_AUROCs = [i[0] for i in diagonals_mapp["individual"]["AUROC"]]
    individ_AUROC_errs = [i[1] for i in diagonals_mapp["individual"]["AUROC"]]
    
    baseline_AUPRC_mapp = pickle.load(open("pickles/random_AUPRC_baseline.pkl", "rb"))
    amyloid_classes = ["cored", "diffuse", "CAA"]
    rand_cons_AUPRCs = [baseline_AUPRC_mapp["consensus"][amyloid_class][0] for amyloid_class in amyloid_classes]
    rand_cons_AUPRC_errs = [baseline_AUPRC_mapp["consensus"][amyloid_class][1] for amyloid_class in amyloid_classes]
    rand_individ_AUPRCs =  [baseline_AUPRC_mapp["individual"][amyloid_class][0] for amyloid_class in amyloid_classes]
    rand_individ_AUPRC_errs =  [baseline_AUPRC_mapp["individual"][amyloid_class][1] for amyloid_class in amyloid_classes]
    rand_AUROCs = [0.5, 0.5, 0.5]

    print("cons AUPRCs", cons_AUPRCs)
    print("cons AUPRCs error", cons_AUPRC_errs)
    print("individual AUPRCs", individ_AUPRCs)
    print("individual AUPRC_errs", individ_AUPRC_errs)
    print("differences", np.array(cons_AUPRCs) - np.array(individ_AUPRCs))
    print("cons AUROCs", cons_AUROCs)
    print("cons AUROCs error", cons_AUROC_errs)
    print("individual AUROCs", individ_AUROCs)
    print("individual AUROC_errs", individ_AUROC_errs)
    print("differences", np.array(cons_AUROCs) - np.array(individ_AUROCs))
    fig, ax = plt.subplots()
    width = .25
    classes = ["Cored", "Diffuse", "CAA"]
    x = np.arange(3)
    xlabels = ["null"] + classes
    ax.bar(x, individ_AUPRCs, width, yerr=individ_AUPRC_errs, capsize=3, ecolor="grey", color="blue", label="Experts AUPRC", zorder=3)
    ax.bar(x, cons_AUPRCs, width,  yerr=cons_AUPRC_errs, capsize=3, color="purple", label="Consensus AUPRC")
    ax.bar(x, rand_individ_AUPRCs, width, yerr=rand_individ_AUPRC_errs, capsize=3, ecolor="white", color="grey", label="Random AUPRC", zorder=4)
    ##random individ and random consensus are equal to two decimal places, just plot one
    # ax.bar(x, rand_cons_AUPRCs, width, yerr=rand_cons_AUPRC_errs, capsize=3, ecolor="white", color="grey", label="Random Consensus AUPRC", zorder=4)
    ax.bar(x + width, individ_AUROCs, width, yerr=individ_AUROC_errs, capsize=3, ecolor="grey", color="gold", label="Experts AUROC",zorder=3)
    ax.bar(x + width, cons_AUROCs, width, yerr=cons_AUROC_errs, capsize=3, color="goldenrod", label="Consensus AUROC")
    ax.bar(x + width, rand_AUROCs, width, capsize=3, color="lightgrey", label="Random AUROC", zorder=4)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.80])
    ax.legend(loc='upper right', fontsize=9, bbox_to_anchor=(1, 1.52))
    plt.gcf().subplots_adjust(top=.69)
    ax.set_ylim((0,1.02))
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.set_xticklabels(xlabels,fontsize=10)
    ax.set_ylabel("Score")
    y_vals = ax.get_yticks()
    ax.set_yticklabels(['{:,.0%}'.format(y) for y in y_vals])
    plt.title("Average Experts and Consensus Models Scores")  
    plt.savefig("figures/avg_AURPC_AUROC_scores_stacked.jpg", dpi=300)

def plotConsensusGainsHeatMap(comparison="model", eval_set="test_set", exclude_amateurs=True, consensus_of_2_only=False):
    """
    Plots a 3x3 grid showing the avg performance gain of using consensus over individuals, for either comparison == "model" or comparison == "truth"
    COMPARISON: "model" or "truth"
    EVAL_SET: "test_set" or "val_set"
    EXCLUDE_AMATEURS: whether we want to exclude the novice models and annotation sets
    CONSENSUS_OF_2_ONLY: whether we want to just use the consensus of 2 model and annotation set to represent the consensus results
    y-axis:
        if COMPARISON=="model": spaces = ["Self Benchmarks", "Consensus Benchmarks", "Individual Benchmarks", "All Benchmarks"]
        else: spaces = ["Self Models", "Consensus Models", "Individual Models", "All Models"]
    x-axis: amyloid_classes = ["cored", "diffuse", "CAA"]
    """
    if comparison == "model":
        spaces = ["Self Benchmarks", "Consensus Benchmarks", "Individual Benchmarks", "All Benchmarks"]
    else:
        spaces = ["Self Models", "Consensus Models", "Individual-Expert Models", "All Models"]
    classes = ["cored", "diffuse", "CAA"]
    grid, z_grid, p_value_grid, intervals_grid = [], [], [], []
    for space in spaces:
        l, z_list, p_list, intervals_list = [], [], [], []
        for c in classes:
            exclude_consensus = True if "Individual" in space else False
            exclude_individuals = True if "Consensus" in space else False
            diagonal_only = True if "Self" in space else False
            avg_consensus, std_consensus, sample_size_consensus = pickle.load(open("pickles/comparison_consensus_vs_individual_consensus_stats_{}_{}_{}_exclude_consensus_{}_exclude_amateurs_{}_diagonal_only_{}_consensus_of_2_only_{}_exclude_individuals_{}.pk".format(comparison, eval_set, c, exclude_consensus, exclude_amateurs, diagonal_only, consensus_of_2_only, exclude_individuals), "rb"))
            avg_individ, std_individ , sample_size_individual= pickle.load(open("pickles/comparison_consensus_vs_individual_individual_stats_{}_{}_{}_exclude_consensus_{}_exclude_amateurs_{}_diagonal_only_{}_consensus_of_2_only_{}_exclude_individuals_{}.pk".format(comparison, eval_set, c, exclude_consensus, exclude_amateurs, diagonal_only, consensus_of_2_only, exclude_individuals), "rb"))
            l.append(avg_consensus - avg_individ)
            z_val = (avg_consensus - avg_individ) / float( np.sqrt( ((std_consensus**2) / float(sample_size_consensus)) + ((std_individ**2) / float(sample_size_individual)) )) 
            z_list.append(z_val)
            cdf_one_sided = scipy.stats.norm.cdf(z_val) 
            cdf_two_sided = (scipy.stats.norm.cdf(z_val) * 2) - 1 
            p_val = 1 - cdf_one_sided ##we want the one sided with Ho: means are equal, alternative: consensus > individuals
            p_list.append(p_val)
            ##calculate 95% confidence interval for difference of two means
            standard_pooled = np.sqrt((((sample_size_individual - 1) * std_individ**2) + (((sample_size_consensus - 1) * std_consensus**2))) / float(sample_size_individual + sample_size_consensus - 2))
            confidence_width = 1.96 * standard_pooled * np.sqrt((1/float(sample_size_individual)) + (1/float(sample_size_consensus))) #z for 95% confidence 
            interval = ((avg_consensus - avg_individ) - confidence_width, (avg_consensus - avg_individ) + confidence_width) 
            intervals_list.append(interval)
        grid.append(l)
        z_grid.append(z_list)
        p_value_grid.append(p_list)
        intervals_grid.append(intervals_list)
    grid = np.array(grid)
    z_grid = np.array(z_grid)
    p_value_grid = np.array(p_value_grid)
    intervals_grid = np.array(intervals_grid)
    print("grid: ",grid)
    ##make heatmap graph
    if comparison == "model":
        graph_compare = "Models"
        converse = "Benchmarks"
    if comparison == "truth":
        graph_compare = "Benchmarks"
        converse = "Models"
    graph_classes = ["Cored", "Diffuse", "CAA"]
    fig, ax = plt.subplots()
    im = ax.imshow(grid,vmin=-.15, vmax=.15)
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(spaces)))
    if consensus_of_2_only:
        ax.set_title("Performance Gains of Consensus-of-Two {} over Individuals".format(graph_compare[0:-1]), fontsize=12, y=1.08) #set y = 1.08 because title too close to figure
    else:
        ax.set_title("Performance Gains of Consensus {} over Individuals".format(graph_compare), fontsize=12, y=1.08) #set y = 1.08 because title too close to figure
    space_averages = {sp: 0 for sp in spaces}
    for i in range(len(spaces)):
        space_sum = 0
        for j in range(len(classes)):
            col_max = max(grid[:,j])
            if grid[i,j] > 0:
                sign = "+"
            else: 
                sign = ""
            if grid[i,j] == col_max:
                weight = "bold"
                fontcolor = "black"
            else:
                weight = "normal"
                fontcolor = "black"
            if i == 0: #self benchmarks / top row, does not meet large N criteria, so exclude p-value
                text = ax.text(j, i, "{}{:.1%}".format(sign, grid[i,j]),  ha="center", va="center", color=fontcolor, fontsize=10, weight=weight)
            else:
                # text = ax.text(j, i, "{}{:.1%}\np={:.1E}".format(sign, grid[i,j], Decimal(p_value_grid[i][j])),  ha="center", va="center", color=fontcolor, fontsize=10, weight=weight)
                text = ax.text(j, i, "{}{:.1%}\np={:.1E}\n ({:.1%},{:.1%})".format(sign, grid[i,j], Decimal(p_value_grid[i][j]), intervals_grid[i,j][0], intervals_grid[i,j][1]),  ha="center", va="center", color=fontcolor, fontsize=8.5, weight=weight)
            space_sum += grid[i,j]
        space_averages[spaces[i]] = space_sum /float(len(classes))
    ax.set_xticklabels(graph_classes,fontsize=10)
    ##annotate yaxis with row averages also
    spaces = ["{}\n(row avg={})".format(space, "{:.2%}".format(space_averages[space])) for space in spaces]
    ax.set_yticklabels(spaces,fontsize=10)
    ax.set_ylim(len(spaces)-0.5, -0.5)
    ax.set_ylabel("{} Considered".format(converse), fontsize=10)
    cbar = ax.figure.colorbar(im, ax=ax)
    plt.gcf().subplots_adjust(left=.25, right=.88) #need to make room for ylabel, default values are left = 0.125, right = 0.9, bottom = 0.1, top = 0.9
    plt.savefig("figures/consensus_performance_gains_heat_map_comparison_{}_eval_set_{}_exclude_amateurs_{}_consensus_of_2_only_{}.png".format(comparison, eval_set, exclude_amateurs, consensus_of_2_only), dpi=300)

def plotNoviceAndConsensusOf2Stats(amateur="UG1"):
    """
    Plots the SSIM on y-axis, pixel thresholds on x-axis for each of the three amyloid beta classes
    """
    amyloid_to_metrics_dict = pickle.load(open("pickles/CAM_amyloid_to_metrics_dict_{}.pk".format(amateur), "rb"))
    thresholds = list(np.arange(0,260, 15))  #uniform axis with more spaced out thresholds
    thresholds.remove(0)
    ##make SSIM plot with all 3 classes
    fig, ax = plt.subplots()    
    x = np.arange(len(thresholds))
    xlabels = [""] + thresholds
    ax.errorbar(x, amyloid_to_metrics_dict[0]["s"], color="blue", label="Cored")
    ax.errorbar(x, amyloid_to_metrics_dict[1]["s"], color="green", label="Diffuse")
    ax.errorbar(x, amyloid_to_metrics_dict[2]["s"], color="gold", label="CAA")
    plt.fill_between(x, np.array(amyloid_to_metrics_dict[0]["s"]) - np.array(amyloid_to_metrics_dict[0]["s_err"]), np.array(amyloid_to_metrics_dict[0]["s"]) + np.array(amyloid_to_metrics_dict[0]["s_err"]), color='blue', alpha=0.05)
    plt.fill_between(x, np.array(amyloid_to_metrics_dict[1]["s"]) - np.array(amyloid_to_metrics_dict[1]["s_err"]), np.array(amyloid_to_metrics_dict[1]["s"]) + np.array(amyloid_to_metrics_dict[1]["s_err"]), color='green', alpha=0.05)
    plt.fill_between(x, np.array(amyloid_to_metrics_dict[2]["s"]) - np.array(amyloid_to_metrics_dict[2]["s_err"]), np.array(amyloid_to_metrics_dict[2]["s"]) + np.array(amyloid_to_metrics_dict[2]["s_err"]), color='gold', alpha=0.10)
    ax.set_ylabel("SSIM", fontsize=12)
    plt.title("Similarity of Saliency Between {} and Consensus-of-Two".format(amateur), fontsize=13, y = 1.02)
    ax.xaxis.set_major_locator(plt.MaxNLocator(len(thresholds) + 1))
    ax.set_xticklabels(xlabels,fontsize=12)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
    y_vals = ax.get_yticks()
    ax.set_yticklabels(['{:,.0%}'.format(x) for x in y_vals], fontsize=12)
    ax.set_ylim((0,1))
    ax.set_xlabel("Pixel Threshold", fontsize=12)
    #Shrink current axis and place legend outside plot, top right corner 
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.85])
    ax.legend(loc='upper right', fontsize=10, bbox_to_anchor=(1, 1.35))
    plt.gcf().subplots_adjust(bottom=0.13, top=.76) #default: left = 0.125, right = 0.9, bottom = 0.1, top = 0.9
    plt.savefig("figures/CAM_SSIM_scores_{}.png".format(amateur), dpi=300)
   
def plotNoviceWithConsensusOf2Difference(amateur="UG1", amyloid_class=None):
    """
    Plots the pixel activations comparing a novice CAM with a consensus-of-two CAM, for a specified AMYLOID_CLASS in (0,1,2)
    """
    granular_thresholds = list(np.arange(0,20, 2)) + list(np.arange(20, 260, 25)) + [255] #more granular search space
    larger_thresholds = list(np.arange(0,260, 15))  #uniform axis with more spaced out thresholds
    thresh_dict = pickle.load(open("pickles/CAM_threshold_stats_{}_{}.pkl".format(amateur, amyloid_class), "rb"))
    for graph_type in ["exclude_agreements", "keep_agreements"]:
        fig, ax = plt.subplots()
        if graph_type == "exclude_agreements":
            thresholds = granular_thresholds
        else:
            thresholds = larger_thresholds
        x = np.arange(len(thresholds))
        xlabels = [""] + thresholds
        A = [thresh_dict[t]["A"][0] for t in thresholds]
        A_err = [thresh_dict[t]["A"][1] for t in thresholds]
        B = [thresh_dict[t]["B"][0] for t in thresholds]
        B_err = [thresh_dict[t]["B"][1] for t in thresholds]
        C = [thresh_dict[t]["C"][0] for t in thresholds]
        C_err = [thresh_dict[t]["C"][1] for t in thresholds]
        ax.errorbar(x, A, color="gold", label="on novice, off consensus")
        plt.fill_between(x, np.array(A) - np.array(A_err), np.array(A) + np.array(A_err), color='gold', alpha=0.10)
        ax.errorbar(x, C, color="blue", label="off novice, on consensus")   
        plt.fill_between(x, np.array(C) - np.array(C_err), np.array(C) + np.array(C_err), color='blue', alpha=0.05)
        if graph_type == "keep_agreements":
            ax.errorbar(x, B, color="green", label="novice and consensus agree")
            plt.fill_between(x, np.array(B) - np.array(B_err), np.array(B) + np.array(B_err), color='green', alpha=0.05)
        ax.xaxis.set_major_locator(plt.MaxNLocator(len(thresholds) + 1))
        xfontsize, yfontsize, xrotation = 12, 12, 45
        if graph_type == "keep_agreements":
            ax.set_ylim((0,1))
        else:
            ax.set_ylim((0,.30))   
        yvals = ax.get_yticks()
        ax.set_yticklabels(['{:,.0%}'.format(x) for x in yvals], fontsize=yfontsize)
        ax.set_xticklabels(xlabels,fontsize=xfontsize)
        plt.setp(ax.get_xticklabels(), rotation=xrotation, ha="right",rotation_mode="anchor")
        am_types = {0: "Cored", 1:"Diffuse", 2:"CAA"}
        ax.set_xlabel("Pixel Threshold", fontsize=12)
        ax.set_ylabel("Proportion", fontsize=12)  
        plt.title("{}".format(am_types[amyloid_class]), fontsize=14)
        #Shrink current axis and place legend outside plot, top right corner 
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width, box.height * 0.85])
        ax.legend(loc='upper left', fontsize=12, bbox_to_anchor=(-0.012, 1.39))
        plt.gcf().subplots_adjust(bottom=0.13, top=.76) #default: left = 0.125, right = 0.9, bottom = 0.1, top = 0.9
        # plt.tight_layout()
        if graph_type == "keep_agreements":
            plt.savefig("figures/CAM_overlap_exclude_{}_{}_{}.png".format(graph_type, amateur, amyloid_class), dpi=300)
        else:
            plt.savefig("figures/CAM_overlap_exclude_{}_{}_{}.png".format(graph_type, amateur, amyloid_class), bbox_inches='tight', dpi=300)

def plotSubsetPercentageConsensusOf2WithAmateur(amateur="UG1", amyloid_class=None):
    """
    Plots a graph comparing 1) how much of a subset the novice CAM is of the consensus-of-two CAM vs
    2) how much of a subset the consensus-of-two CAM is of the novice CAM for a specified AMYLOID_CLASS in (0,1,2)
    """
    granular_thresholds = list(np.arange(0,20, 2)) + list(np.arange(20, 260, 25)) + [255] #more granular search space
    thresh_dict = pickle.load(open("pickles/CAM_subset_dict_{}_{}.pk".format(amateur, amyloid_class), "rb"))
    fig, ax = plt.subplots()
    thresholds = granular_thresholds 
    x = np.arange(len(thresholds))
    xlabels = [""] + thresholds  
    ax.errorbar(x, [thresh_dict["consensus"][t][0] for t in thresholds], color="orange", label="percentage of consensus found in novice") 
    plt.fill_between(x, np.array([thresh_dict["consensus"][t][0] for t in thresholds]) - np.array([thresh_dict["consensus"][t][1] for t in thresholds]), np.array([thresh_dict["consensus"][t][0] for t in thresholds]) + np.array([thresh_dict["consensus"][t][1] for t in thresholds]), color='orange', alpha=0.10)
    ax.errorbar(x, [thresh_dict["amateur"][t][0] for t in thresholds],  color="darkcyan", label="percentage of novice found in consensus") 
    plt.fill_between(x, np.array([thresh_dict["amateur"][t][0] for t in thresholds]) - np.array([thresh_dict["amateur"][t][1] for t in thresholds]), np.array([thresh_dict["amateur"][t][0] for t in thresholds]) + np.array([thresh_dict["amateur"][t][1] for t in thresholds]), color='darkcyan', alpha=0.05)
    am_types = {0: "Cored", 1:"Diffuse", 2:"CAA"}    
    plt.title("{}".format(am_types[amyloid_class]), fontsize=14)
    ax.xaxis.set_major_locator(plt.MaxNLocator(len(thresholds) + 1))
    yvals = ax.get_yticks()
    ax.set_yticklabels(['{:,.0%}'.format(x) for x in yvals], fontsize=12)
    ax.set_xticklabels(xlabels,fontsize=12)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")  
    ax.set_xlabel("Pixel Threshold", fontsize=12)
    ax.set_ylabel("Percentage", fontsize=12)  
    #Shrink current axis and place legend outside plot, top right corner 
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.85])
    ax.legend(loc='upper left', fontsize=12, bbox_to_anchor=(-.012, 1.39))
    plt.gcf().subplots_adjust(bottom=0.135, top=.76) #default: left = 0.125, right = 0.9, bottom = 0.1, top = 0.9
    plt.savefig("figures/CAM_subset_percentages_{}_{}.png".format(amateur, amyloid_class), dpi=300)

def plotEnsembleSuperiority(random_subnet=False, multiple_subnets=False):
    """
    Plots a differential bar graph for ensemble superiority
    RANDOM_SUBNET: whether or not we're analyzing ensembles with a single random subnet included
    MULTIPLE_RANDOMS: whether or not we're analyzing ensembles with multiple random subnets included
    Requires "ensemble_superiority_difference_map_random_subnet_{}_multiple_subnets_{}.pkl"
    """
    USERS = ['NP1', 'NP2', 'NP3', 'NP4', 'NP5']
    difference_map = pickle.load(open("pickles/ensemble_superiority_difference_map_random_subnet_{}_multiple_subnets_{}.pkl".format(random_subnet, multiple_subnets), "rb"))
    fig, ax = plt.subplots()
    width = .27
    am_classes = [0,1,2]
    consensus_differentials = [np.mean(difference_map[am_class]["consensus"]) for am_class in am_classes]
    consensus_differentials_stds = [np.std(difference_map[am_class]["consensus"]) for am_class in am_classes]    
    individuals_differentials = [np.mean(difference_map[am_class]["individual"]) for am_class in am_classes]
    individuals_differentials_stds = [np.std(difference_map[am_class]["individual"]) for am_class in am_classes]    
    x = np.arange(len(am_classes))
    ax.bar(x, consensus_differentials, width,  yerr=consensus_differentials_stds, capsize=3, color="gold", label="Consensus Ensembles - Consensus CNNs")
    ax.bar(x + width, individuals_differentials, width,  yerr=individuals_differentials_stds, capsize=3, color="blue", label="Expert Ensembles - Individual-Expert CNNs")
    xlabels = ["null", "Cored", "Diffuse", "CAA"]
    ax.set_xticklabels(xlabels,fontsize=10)
    ax.set_ylabel("Ensemble - Single CNN Performance (AUPRC)", fontsize=10)
    plt.title("Ensemble Performance Gains over Single CNNs", fontsize=12)
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.85])
    ax.legend(loc='upper right', fontsize=10, bbox_to_anchor=(1.013, 1.3))
    plt.gcf().subplots_adjust(bottom=0.13, top=.76)
    plt.savefig("figures/ensemblesVsSingletons_differentials_rand_subnet_{}_multiple_subnets_{}.jpg".format(random_subnet, multiple_subnets), dpi=300)

def plotEnsembleDiffHistograms(single_random=False, multiple_subnets=False): 
    """
    Plots the histogram of performance differences between 1) normal ensembles and 2) ensembles with a single random subnet, or ensembles with multiple random subnets
    SINGLE_RANDOM: whether or not we're analyzing ensembles with a single random subnet included
    MULTIPLE_SUBNETS: whether or not we're analyzing ensembles with multiple random subnets included
    """
    cored = pickle.load(open("pickles/ensemble_difference_values_0_random_{}_multiple_{}.pkl".format(single_random, multiple_subnets), "rb"))
    diffuse = pickle.load(open("pickles/ensemble_difference_values_1_random_{}_multiple_{}.pkl".format(single_random, multiple_subnets), "rb"))
    CAA = pickle.load(open("pickles/ensemble_difference_values_2_random_{}_multiple_{}.pkl".format(single_random, multiple_subnets), "rb"))
    X = np.transpose(np.array([cored, diffuse, CAA]))
    print(X.shape)
    fig, ax = plt.subplots()
    colors = ["maroon", "bisque", "darkorange"]
    n, bins, patches = ax.hist(X, 10, histtype='barstacked', stacked=True, color=colors, label=["Cored", "Diffuse", "CAA"])# stacked bars
    n = np.array(n)
    sum_of_rows = n.sum(axis=1)
    normalized_n = n / sum_of_rows[:, np.newaxis]
    print(len(normalized_n), normalized_n)
    print("bins", len(bins), bins)
    new_x = [] ##midpoint between each bin coordinate
    for i in range(0, len(bins) - 1):
        new_x.append(bins[i] + (bins[i+1] - bins[i] / float(2)))
    print(len(new_x), new_x)
    print(normalized_n.shape)
    fig, ax = plt.subplots()
    amyloid_classes = ["Cored", "Diffuse", "CAA"]
    width = new_x[1] - new_x[0]
    for i in range(0, 3):
        ax.bar(new_x, normalized_n[i], width=width, color='None', edgecolor=colors[i], label=amyloid_classes[i])
    ax.set_ylim(0,1) 
    y_vals = ax.get_yticks()
    ax.set_yticklabels(['{:,.0%}'.format(y) for y in y_vals])  
    ax.set_ylabel("Frequency", fontsize=10)
    if single_random:
        ax.set_xlabel("|AUPRC Difference|\nBetween Ensemble and Ensemble with Random Labeler", fontsize=10)
    else:
        ax.set_xlabel("|AUPRC Difference|\nBetween Ensemble and Ensemble with Multiple Random Labelers", fontsize=10)
    ax.set_title("Ensemble AUPRC Differences Over All Benchmarks", fontsize=12, y = 1.03)     
    ##top right corner 
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.85])
    ax.legend(loc='upper right', fontsize=9, bbox_to_anchor=(1, 1.35))
    plt.gcf().subplots_adjust(bottom=.17, top=.76)
    if single_random:
        plt.savefig("figures/histogram_effect_of_random_on_ensemble_class.png", dpi=300)
    if multiple_subnets:
        plt.savefig("figures/histogram_effect_of_multiple_random_on_ensemble_class.png", dpi=300)

def plotNegativeFlagNotSure(phase="phase1"):
    """
    Plots the supplemental figure breakdown of negative, flag, and not sure
    Relies on method getCountsOfNegativeFlagUnsure
    """
    USERS = ['NP1', 'NP2', 'NP3', 'NP4', 'NP5']
    categories = ["negative", "flag", "notsure", "negatives_with_pos", "flags_with_pos", "notsures_with_pos", "no_pos_marked"]
    subcategories = ["negative", "flag", "notsure"]
    subcategories_2 = ["negatives_with_pos", "flags_with_pos", "notsures_with_pos"]
    lists_dict = pickle.load(open("pickles/negativeFlagNotSure_lists_dict_{}.pkl".format(phase), "rb"))
    counts_dict = pickle.load(open("pickles/negativeFlagNotSure_counts_dict_{}.pkl".format(phase), "rb"))
    ## edits for formating graph 
    lists_dict["not sure"] = lists_dict["notsure"]
    lists_dict["not sures_with_pos"] = lists_dict["notsures_with_pos"]
    del lists_dict["notsure"]
    del lists_dict["notsures_with_pos"]
    subcategories.append("not sure")
    subcategories.remove("notsure")
    subcategories_2.append("not sures_with_pos")
    subcategories_2.remove("notsures_with_pos")

    # ##anonymize names remove NP identifiers further for these figures
    greek_key = pickle.load(open("/srv/home/dwong/privateRepoForConsensusPaper/greek_key.pkl", "rb")) ##key is private
    xlabels = [greek_key[x] for x in USERS]
    for key in lists_dict:
        null, lists_dict[key] = zip(*sorted(zip(xlabels, lists_dict[key]))) ##if sort multiple times, will lose order, so solution is to sort and assign xlabels at the end, after all sorting of dictionary lists is completed
    xlabels = sorted(xlabels)

    ##plot for negative, flag, and not sure
    fig, ax = plt.subplots()
    x = np.array(list(range(0, len(USERS))))
    width = .18
    offset = 0
    for cat in subcategories:    
        ax.bar(x + offset, lists_dict[cat], width=width, label=capitalizeEachWord(cat))
        offset += width 
    plt.xticks(x, xlabels, fontsize=10)
    y_vals = ax.get_yticks()
    ax.set_yticklabels(['{}'.format(int(y)) for y in y_vals], fontsize=10)  
    #Shrink current axis and place legend outside plot top right corner 
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.85])
    ax.legend(loc='upper right', fontsize=9, bbox_to_anchor=(1, 1.40))
    plt.gcf().subplots_adjust(bottom=0.20, top=.76)   
    ax.set_ylabel("Count", fontsize=12)
    if phase == "phase1":
        plt.title("Phase One: Counts of Negative, Flag, and Not Sure", fontsize=14)
    else:
        plt.title("Phase Two: Counts of Negative, Flag, and Not Sure", fontsize=14)
    plt.savefig("figures/flag_annotation_stats_{}.png".format(phase), dpi=300) 
    
    ##plot percentages of marked categories AND marked positive
    fig, ax = plt.subplots()
    offset = 0
    for cat2 in subcategories_2:
        corresponding_cat = [x for x in subcategories if x in cat2][0]
        cat2_list = np.array(lists_dict[cat2])
        cat1_list = np.array(lists_dict[corresponding_cat])
        ratio = cat2_list / cat1_list
        ax.bar(x + offset, ratio, width=width, label=capitalizeEachWord(corresponding_cat))
        offset += width
    plt.xticks(x, xlabels, fontsize=10)
    y_vals = ax.get_yticks()
    ax.set_yticklabels(['{:.0%}'.format(y) for y in y_vals], fontsize=10)  
    #Shrink current axis and place legend outside plot top right corner 
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.85])
    ax.legend(loc='upper right', fontsize=9, bbox_to_anchor=(1, 1.45))
    plt.gcf().subplots_adjust(bottom=0.15, top=.74)   
    ax.set_ylabel("Percentage", fontsize=12)
    if phase == "phase1":
        plt.title("Phase One: Percentages of Negative, Flag, and Not Sure\nMarked with a Positive Amyloid Label", fontsize=14, y=1.03)
    else:
        plt.title("Phase Two: Percentages of Negative, Flag, and Not Sure\nMarked with a Positive Amyloid Label", fontsize=14, y=1.03)

    plt.savefig("figures/flag_annotation_stats_{}_subset_marked_pos.png".format(phase), dpi=300)  

    ##plot fraction of annotations that were considered negative (i.e. no positive annotations)
    lists_dict['no_pos_marked'] = [x / float(20099)  for x in lists_dict['no_pos_marked']]
    fig, ax = plt.subplots()
    ax.bar(x, lists_dict['no_pos_marked'], width=width, color="slategrey")
    plt.xticks(x, xlabels, fontsize=10)
    ax.set_ylim((0,1))
    y_vals = ax.get_yticks()
    ax.set_yticklabels(['{:.0%}'.format(y) for y in y_vals], fontsize=10) 
    ax.set_ylabel("Percentage", fontsize=12)
    if phase == "phase1":
        plt.title("Phase One: Percentage of Annotations\nWithout a Positive Amyloid Label", fontsize=14, y=1.03)
    else:
        plt.title("Phase Two: Percentage of Annotations \nWithout a Positive Amyloid Label", fontsize=14, y=1.03)
    plt.savefig("figures/flag_annotation_stats_{}_no_pos_marked.png".format(phase), dpi=300)  
 
def correlateInterraterWithAverages(am_type=None, eval_set="test_set", exclude_amateurs=False):
    """
    correlates interrater kappa with grid averages
    exclude_consensus and exclude_amateurs is passed simply to keep track of if we're doing this calculation with the amateurs or not, and is used for reading the right pickles and for naming the saved plots

    """
    am_types = {0: "cored", 1: "diffuse", 2:"CAA"}
    # kappas = pickle.load(open("pickles/avg_sorted_user_kappa_{}.pk".format(am_types[am_type]), "rb"))
    kappas = pickle.load(open("pickles/phase1_avg_sorted_user_kappa_{}_exclude_novices_{}_test_set_only_False.pk".format(am_types[am_type], exclude_amateurs), "rb"))
    columns = pickle.load(open("pickles/avg_sorted_user_column_{}_exclude_self_True_exclude_consensus_True_exclude_amateurs_{}.pk".format(am_types[am_type], exclude_amateurs), "rb"))
    rows = pickle.load(open("pickles/avg_sorted_user_row_{}_exclude_self_True_exclude_consensus_True_exclude_amateurs_{}.pk".format(am_types[am_type], exclude_amateurs), "rb"))
    if exclude_amateurs:
        kappas = {k for k in kappas if "UG1" not in k[0] and "UG2" not in k[0]}
    ##clean up names and get rid of consensus entries 
    cleaned_columns = []
    for x in columns:
        if "thresholding" not in x[0]:
            cleaned_columns.append((getUser(x[0]), x[1]))
    cleaned_rows = []
    for x in rows:
        if "thresholding" not in x[0] and "ensemble" not in x[0]:
            cleaned_rows.append((getUser(x[0]), x[1]))
    columns = cleaned_columns
    rows = cleaned_rows
    ##sort
    kappas = sorted(kappas, key=lambda tup: tup[1])
    columns = sorted(columns, key=lambda tup: tup[1])
    rows = sorted(rows, key=lambda tup: tup[1])
    ##find correlation between avg value performance values, and also kappa values:
    axis = "columns"
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress([kappa[1] for kappa in kappas], [float(column[1]) for column in columns])
    new_list = np.array([float(column[1]) for column in columns])
    ##make plot
    og_list = np.array([kappa[1] for kappa in kappas])
    fig, ax = plt.subplots()
    for i, label in enumerate([kappa[0] for kappa in kappas]): ##enumerate names
        x = og_list[i]
        y = new_list[i]
        ax.scatter(x, y, color="#1f77b4")
        pltLabel = label
        if am_type == 1: ##diffuse too scrunched together
            if i == 1:
                x -= .01
                y -= .001
            ax.text(x -.01, y+0.001, pltLabel, fontsize=10)
        else:
            ax.text(x+0.0, y+0.0, pltLabel, fontsize=10)
    ax.set_ylabel("Performance (AUPRC)",fontsize=10)
    ax.set_xlabel("Interrater Agreement (kappa)",fontsize=10)
    am_types = {0: "Cored", 1: "Diffuse", 2:"CAA"}
    ax.set_title(am_types[am_type], y=1.01, fontsize=13)
    ax.plot(og_list, slope*og_list + intercept, color="r", label="best fit line, $R^2$ = {}".format(str(round(r_value**2, 2))[0:4])) #best fit line
    ##shrink axis and put legend outside
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.84])
    ax.legend(loc='upper right', fontsize=11, bbox_to_anchor=(1, 1.25))
    plt.gcf().subplots_adjust(bottom=0.13, top=.80) #default: left = 0.125, right = 0.9, bottom = 0.1, top = 0.9
    plt.savefig("figures/{}_interrater_grid_performance_avg_{}_{}_exclude_amateurs_{}.png".format(eval_set, am_type, axis, exclude_amateurs), dpi=300)
    plt.cla()
    plt.clf()

def plotPerformanceByStainType():
    """
    Plots and saves a bar chart of performances by stain, amyloid class, and model 
    """
    margin = 0
    width = .20
    colors = {"UTSW": 'blue', "UCD":'gold', "cw": 'red'}
    antibody_map = {"UTSW": "6E10", "UCD": "4G8", "cw":"NAB228"}
    amyloid_dict = {0: "Cored", 1: "Diffuse", 2: "CAA"}
    counts_dict = pickle.load(open("pickles/stain_type_counts_dict_test_set.pkl", "rb"))
    for eval_metric in ["AUPRC", "AUROC"]:
        if eval_metric == "AUPRC":
            eval_index = 0
        if eval_metric == "AUROC":
            eval_index = 1
        #key model name, key: stain type, key: amyloid class, value: AUPRC    
        results = pickle.load(open("pickles/compare_stain_types_images.pkl", "rb"))
        for amyloid_class in [0,1,2]:
            fig, ax = plt.subplots()
            for stain_type in ["UTSW", "UCD", "cw"]: 
                y = []
                models = results.keys()
                x = np.arange(0, len(models))
                for model in results.keys():
                    y.append(results[model][stain_type][amyloid_class][eval_index])
                ax.bar(x + margin, y, width=width, label= antibody_map[stain_type] + ", $\it{n}$=" + str(counts_dict[stain_type]), color=colors[stain_type])
                margin += width 
            xlabels = ['null'] + list(models)
            xlabels = ["C2" if x == "thresholding_2" else x for x in xlabels]
            ax.set_xticklabels(xlabels,fontsize=10)
            ax.set_xlabel("Model", fontsize=10)
            ax.set_ylabel(eval_metric, fontsize=10)
            ax.set_ylim((0,1))
            y_vals = ax.get_yticks()
            ax.set_yticklabels(['{:,.0%}'.format(y_val) for y_val in y_vals], fontname="Times New Roman")
            plt.title("{}s by Stain Type\n{}".format(eval_metric, amyloid_dict[amyloid_class]), fontsize=14, y = 1.01)
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width, box.height * 0.85])
            ax.legend(loc='upper right', fontsize=9, bbox_to_anchor=(1, 1.38))
            plt.gcf().subplots_adjust(top=.76)   
            plt.savefig("figures/{}_by_stain_type_{}.png".format(eval_metric, amyloid_dict[amyloid_class]), dpi=300)

def plotEnsembleWeights(eval_random_subnets=False, eval_amateurs=False,eval_multiple=False):
    """
    Plots the sparse affine weights after a (custom) softmax operation
    """
    USERS = ['UG1', 'UG2', 'NP1', 'NP2', 'NP3', 'NP4', 'NP5']
    if eval_random_subnets:
        USERS.append('random1')
    elif eval_multiple:
        USERS += ['random1','random2','random3','random4','random5']
    if eval_amateurs == False:
        USERS = [u for u in USERS if u != "UG1" and u != "UG2"]
    classes = [0,1,2]
    x_coords = [user + "_" + str(c) for user in USERS for c in classes]
    mapp = pickle.load(open("pickles/weights_random_subnet_" + str(eval_random_subnets) +  "_eval_amateurs_" + str(eval_amateurs) + "_multiple_" + str(eval_multiple) + ".p", "rb")) 
    models = list(set([x[0] for x in mapp.keys()]))
    models = customUserSort(models)
    x_coords = customUserSort(x_coords)
    ##now plot the mapp as a 2D grid
    grid = [] #2D grid of x: csv dataset, y: model, value: AUPRC score
    for model_name in models:
        model_results = []
        for x in x_coords:
            model_results.append(float(str(mapp[(model_name, x)])[0:5]))
        grid.append(model_results)
    grid = np.array(grid)
    fig, ax = plt.subplots()
    im = ax.imshow(grid,vmin=0, vmax=0.40)
    ax.set_xticks(np.arange(len(x_coords)))
    ax.set_yticks(np.arange(len(models)))
    # label them with the respective list entries
    models = [x.replace("ensemble_model", "") for x in models]
    models = [x.replace("_multiple_rand_subnets_", "") for x in models]
    x_coords_names = x_coords
    x_coords_names = [x.replace("_0", " Cored") for x in x_coords_names]
    x_coords_names = [x.replace("_1", " Diffuse") for x in x_coords_names]
    x_coords_names = [x.replace("_2", " CAA") for x in x_coords_names]
    x_coords_names = [x.replace("random1", " random") for x in x_coords_names]
    models_names = models
    models_names = [x.replace("_", "") for x in models_names]
    models_names = [x.replace("l2.pkl", "") for x in models_names]
    models_names = [x.replace("ensemblerandomsubnetmodel", "") for x in models_names]
    models_names = [x.replace("allthresholding", "Consensus of ") for x in models_names]
    ax.set_xticklabels(x_coords_names,fontsize=6)
    ax.set_yticklabels(models_names,fontsize=6)
    if eval_random_subnets:
        ax.set_ylabel("Ensemble Model with a Random Subnet",fontsize=8)
    elif eval_multiple:
        ax.set_ylabel("Ensemble Model with Five Random Subnets", fontsize=8)
    else:
        ax.set_ylabel("Ensemble Model", fontsize=8)
    ax.set_xlabel("Neuron")
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    for i in range(len(models)):
        for j in range(len(x_coords)):
            text = ax.text(j, i, str(grid[i, j])[1:4],  ha="center", va="center", color="w", fontsize=6)
    cbar = ax.figure.colorbar(im, ax=ax)
    if eval_random_subnets:
        ax.set_title("Weights of Ensembles with Random Subnet" , fontsize=10)
    elif eval_multiple:
        ax.set_title("Weights of Ensembles with Multiple Random Subnets" , fontsize=10)
    else:
        ax.set_title("Weights of Normal Ensembles" , fontsize=10)
    fig.tight_layout()
    plt.savefig("figures/weight_eval_random_subnets_" + str(eval_random_subnets) + "_multiple_" + str(eval_multiple) + ".png", dpi=300)

def plotStainUserClassCounts(dataset="full"):
    """
    plots and saves bar charts of y = positive annotation count by amyloid class, x = user, and legend = stain
    DATASET either "full" or "test"
    """
    margin = 0
    width = .20
    colors = {"UTSW": 'blue', "UCD":'gold', "cw": 'red'}
    antibody_map = {"UTSW": "6E10", "UCD": "4G8", "cw":"NAB228"}
    amyloid_dict = {0: "Cored", 1: "Diffuse", 2: "CAA"}
    counts_dict = pickle.load(open("pickles/stain_type_counts_dict_test_set.pkl", "rb"))
    USERS = ['UG1', 'UG2', 'NP1', 'NP2', 'NP3', 'NP4', 'NP5', 'thresholding_2']
    stain_user_class_counts_dict = pickle.load(open("pickles/stain_user_class_counts_dict_{}_set.pkl".format(dataset), "rb"))
    for amyloid_class in [0,1,2]:
        fig, ax = plt.subplots()
        for stain_type in ["UTSW", "UCD", "cw"]: 
            y = []
            x = np.arange(0, len(USERS))
            for user in USERS:
                y.append(stain_user_class_counts_dict[user][stain_type][amyloid_class][0])
            if dataset == "test":
                ax.bar(x + margin, y, width=width, label= antibody_map[stain_type] + ", $\it{n}$=" + str(counts_dict[stain_type]), color=colors[stain_type])
            else:
                ax.bar(x + margin, y, width=width, label= antibody_map[stain_type], color=colors[stain_type])

            margin += width 
        xlabels = ['null'] + USERS
        xlabels = ["C2" if x == "thresholding_2" else x for x in xlabels]
        ax.set_xticklabels(xlabels,fontsize=10)
        ax.set_xlabel("Annotator", fontsize=10)
        ax.set_ylabel("Number of Positive Annotations", fontsize=10)
        y_vals = ax.get_yticks()
        plt.title("Positive {} Annotation Counts in {} Dataset".format(amyloid_dict[amyloid_class], dataset.capitalize()), fontsize=14, y = 1.01)
        #Shrink current axis and place legend outside plot top right corner 
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width, box.height * 0.85])
        ax.legend(loc='upper right', fontsize=9, bbox_to_anchor=(1, 1.38))
        plt.gcf().subplots_adjust(top=.76)   
        plt.savefig("figures/positive_counts_by_stain_type_{}_set_{}.png".format(dataset, amyloid_class), dpi=300)

def plotColorNormVsUnnormalized():
    color_norm_dict = pickle.load(open("pickles/compare_color_norm_vs_unnorm.pkl", "rb"))
    users = list(color_norm_dict["color_norm"].keys())
    differences_dict = {user: {0:0, 1:0, 2:0} for user in users}
    for user in users:
        for amyloid_class in color_norm_dict["color_norm"][user]:
            differences_dict[user][amyloid_class] = color_norm_dict["color_norm"][user][amyloid_class] - color_norm_dict["unnorm"][user][amyloid_class] 
    fig, ax = plt.subplots()
    x = np.arange(len(users))
    margin = 0
    width = .20
    colors = ["maroon", "bisque", "darkorange"]
    amyloid_dict = {0: "Cored", 1: "Diffuse", 2:"CAA"}
    for amyloid_class in [0,1,2]:
        scores_list = [differences_dict[user][amyloid_class] for user in users]
        ax.bar(x + margin, scores_list, width=width, label=amyloid_dict[amyloid_class], color=colors[amyloid_class])
        margin += width 
    xlabels = ['null'] + users
    xlabels = ["C2" if x == "thresholding_2" else x for x in xlabels]
    ax.set_xticklabels(xlabels,fontsize=10)
    ax.set_xlabel("Model", fontsize=10)
    ax.set_ylabel("AUPRC Difference (Normalized - Unnormalized)", fontsize=10)
    plt.title("AUPRC Difference over Test Set", fontsize=14, y = 1.01)
    #Shrink current axis and place legend outside plot top right corner 
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.85])
    ax.legend(loc='upper right', fontsize=9, bbox_to_anchor=(1, 1.30))
    plt.gcf().subplots_adjust(top=.76)   
    plt.savefig("figures/comparing_color_norm_vs_unnorm.png", dpi=300)

def plotTrainLossCurve():
    """
    Plots the consensus of 2 fold 3 model train and loss curves
    """
    train_loss = []
    validation_loss = []
    with open("train_loss_curve.txt") as f:
        lines = f.readlines()
        for line in lines:
            if "train Loss" in line:
                train_loss.append(float(line.split("train Loss: ")[1]))
            if "dev Loss" in line:
                validation_loss.append(float(line.split("dev Loss: ")[1]))
    fig, ax = plt.subplots()
    x = list(range(0, len(train_loss)))
    xlabels = ['null'] + list(range(0, len(train_loss)))
    ax.plot(x, train_loss, label = "training loss", color='blue')
    ax.plot(x, validation_loss, label = "validation loss", color='orange')
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("MultiLabel Soft Margin Loss", fontsize=10)
    plt.title("Training and Validation Loss\nfor Consensus-of-Two Model", fontsize=14, y = 1.01)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.85])
    ax.legend(loc='upper right', fontsize=9, bbox_to_anchor=(1, 1.33))
    plt.gcf().subplots_adjust(top=.76)   
    plt.savefig("figures/train_loss_curve_c2.png", dpi=300)












