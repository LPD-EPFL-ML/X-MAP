# -*- coding: utf-8 -*-
"""An auxilary file for xmap."""


def clean_data_pipeline(sc, clean_tool, path_rawdata):
    """a pipeline to clean the data."""
    dataRDD = sc.textFile(path_rawdata, 30)
    parsedRDD = clean_tool.parse_data(dataRDD)
    filteredRDD = clean_tool.filter_data(parsedRDD)
    cleanedRDD = clean_tool.clean_data(filteredRDD).cache()
    partial_data = clean_tool.take_partial_data(cleanedRDD)
    partialRDD = sc.parallelize(partial_data, 30).cache()
    return partialRDD


def split_data_pipeline(sc, split_tool, sourceRDD, targetRDD):
    """a pipeline to clean the data."""
    overlap_userRDD = split_tool.find_overlap_user(sourceRDD, targetRDD)
    overlap_userRDD_bd = sc.broadcast(overlap_userRDD.collect())

    non_overlap_sourceRDD, overlap_sourceRDD, \
        non_overlap_targetRDD, overlap_targetRDD = split_tool.distinguish_data(
            overlap_userRDD_bd, sourceRDD, targetRDD)
    training_dataRDD, test_dataRDD = split_tool.split_data(
        non_overlap_sourceRDD, overlap_sourceRDD,
        non_overlap_targetRDD, overlap_targetRDD)

    training_dataRDD, test_dataRDD \
        = training_dataRDD.cache(), test_dataRDD.cache()
    return training_dataRDD, test_dataRDD


def itembasedsim_pipeline(sc, itemsim_tool, trainRDD):
    """a pipeline to calculate itembased sim."""
    universal_user_info = itemsim_tool.get_universal_user_info(trainRDD)
    user_info = sc.broadcast(universal_user_info.collectAsMap())

    universal_item_info = itemsim_tool.get_universal_item_info(
        trainRDD, user_info)
    item_info = sc.broadcast(universal_item_info.collectAsMap())

    item2item_simRDD = itemsim_tool.calculate_item2item_sim(
        trainRDD, item_info, user_info).cache()
    return item2item_simRDD


def extender_pipeline(
        sc, sqlContext, itemsim_tool, extendsim_tool, item2item_simRDD):
    item2item_simDF = itemsim_tool.build_sim_DF(item2item_simRDD)
    item2item_simDF.registerTempTable("sim_table")
    BB_item_list = sqlContext.sql(
        "SELECT DISTINCT id1 FROM sim_table WHERE label = 1").map(
        lambda line: line.id1).collect()
    BB_item_bd = sc.broadcast(BB_item_list)

    item_simRDD = itemsim_tool.get_item_sim(item2item_simRDD)
    classfied_items = extendsim_tool.find_knn_items(
        item_simRDD, BB_item_bd).cache()

    BB_info, NB_info, knn_BB_bd, knn_NB_bd = extract_siminfo(
        sc, classfied_items)
    BB_info, NB_info = BB_info.cache(), NB_info.cache()

    cross_extended_sim = extendsim_tool.sim_extend(
        BB_info, NB_info, knn_BB_bd, knn_NB_bd)
    final_extended_sim = extendsim_tool.get_final_extension(
        cross_extended_sim).cache()

    return final_extended_sim


def extract_siminfo(sc, classfied_items):
    """broadcast knn item information.
    arg:
        classfied_items: iid, (BB_BB, BB_NB), (NB_BB, NB_NN)
    return:
        knn_BB_bd: {BB iid: {NB iid: (sum, mutu, frac_mutu)}}
        knn_NB_bd: {NB iid: {iid: (sum, mutu, frac_mutu)}}
    """
    BB_info = classfied_items.map(
        lambda line: (line[0], line[1])).filter(
        lambda line: line[1] is not None)

    NB_info = classfied_items.map(
        lambda line: (line[0], line[2])).filter(
        lambda line: line[1] is not None)

    BB_items_knn = BB_info.map(
        lambda line: (line[0], dict(
                (l[0], l[1:]) for l in line[1][0] + line[1][1]))
    ).collectAsMap()

    NB_items_knn = NB_info.map(
        lambda line: (line[0], dict(
                (l[0], l[1:]) for l in line[1][0] + line[1][1]))
    ).collectAsMap()

    knn_BB_bd = sc.broadcast(BB_items_knn)
    knn_NB_bd = sc.broadcast(NB_items_knn)
    return BB_info, NB_info, knn_BB_bd, knn_NB_bd


def private_mapping_pipeline(privatemap_tool, extended_simRDD, private):
    """a pipeline to private map item.
    args:
        private: boolean value.
    """
    if private:
        mappedRDD = privatemap_tool.cross_private_mapping(
            extended_simRDD)
    else:
        mappedRDD = privatemap_tool.cross_nonprivate_mapping(
            extended_simRDD)
    return mappedRDD, map_to_dict(mappedRDD)


def crosssim_pipeline(sc, cross_sim_tool, training_dataRDD, mapped_sim):
    """a pipeline to calculate crosssim in target domain."""
    mapped_sim_dict = map_to_dict(mapped_sim)
    alterEgo_profile = cross_sim_tool.build_alterEgo(
        training_dataRDD, mapped_sim_dict).cache()

    user_based_alterEgo = cross_sim_tool.build_sthbased_profile(
        alterEgo_profile, "user").cache()
    item_based_alterEgo = cross_sim_tool.build_sthbased_profile(
        alterEgo_profile, "item").cache()
    user_based_dict_bd = sc.broadcast(user_based_alterEgo.collectAsMap())
    item_based_dict_bd = sc.broadcast(item_based_alterEgo.collectAsMap())

    user_info = cross_sim_tool.get_info(user_based_alterEgo)
    item_info = cross_sim_tool.get_info(item_based_alterEgo)

    user_info_bd = sc.broadcast(user_info.collectAsMap())
    item_info_bd = sc.broadcast(item_info.collectAsMap())

    targetdomain_sim = cross_sim_tool.calculate_sim(
        item_based_alterEgo, user_based_alterEgo,
        item_info_bd, user_info_bd).cache()

    return user_based_alterEgo, item_based_alterEgo, \
        user_based_dict_bd, item_based_dict_bd, \
        user_info_bd, item_info_bd, targetdomain_sim


def private_policy_pipeline(policy_tool, alterEgo_sim, policy_method):
    """a pipeline that apply private policy on alterEgo profile."""
    if policy_method == "PSA":
        private_neighbor = policy_tool.private_neighbor_selection(
            alterEgo_sim)
        noise_perturbed = policy_tool.noise_perturbation(
            private_neighbor)
        return noise_perturbed
    else:
        no_private_neighbor = policy_tool.nonprivate_neighbor_selection(
            alterEgo_sim)
        no_noise_perturbed = policy_tool.nonnoise_perturbation(
            no_private_neighbor)
        return no_noise_perturbed


def map_to_dict(rdd):
    """For a rdd, map it from list to dict.
    return:
        {source item: target item}
    """
    return dict((line[1], line[0]) for line in rdd.collect())


def convertToList(items, chunk):
    return zip(*[iter(items)]*chunk)
