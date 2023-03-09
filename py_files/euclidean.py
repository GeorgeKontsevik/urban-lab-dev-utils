import geopandas as gpd
import pandas as pd
from time import sleep
from shapely.geometry import mapping
from geopy.distance import distance
from tqdm.auto import tqdm
import numpy as np
from loguru import logger

tqdm.pandas()


class Euclidean:
    def __init__(self):
        self.common_metric_crs = 3857
        self.common_not_metric_crs = 4326


    def _get_coordinates(blocks, current_block, block_id_column) -> tuple:
        coords = blocks.loc[blocks[block_id_column] == current_block, 'coords'].squeeze()
        return coords


    def _get_distances(source, target):
        dist = round(distance(source, target).m, 0)
        return dist


    def _calc_dist(current_block, distance_matrix, blocks, block_ids, block_id_column) -> None:
        print('Осталось блоков: ', len(block_ids), end="\r")
        del block_ids[0]
        
        source = Euclidean._get_coordinates(blocks, current_block, block_id_column)
        targets = np.array(blocks.loc[blocks[block_id_column].isin(block_ids)].coords)
        lst = [Euclidean._get_distances(source, t) for t in targets]
        distance_matrix.loc[block_ids, current_block] = lst

        return distance_matrix


    def calc_blocks_euclidean(self, blocks: gpd.GeoDataFrame, block_id_column: str):
        logger.info('Starting')
        ''' 
        На вход: полигоны кварталов города в формате geojson 
        WARNING: работает оч долго, соре, быстрее не придумал
        '''

        logger.info("Getting blocks' centroids")

        blocks = blocks.to_crs(self.common_metric_crs)
        blocks['centroids'] = blocks['geometry'].centroid
        blocks.drop(columns=['geometry'], inplace=True)
        blocks.rename(columns={'centroids':'geometry'}, inplace=True)
        blocks = gpd.GeoDataFrame(blocks, geometry='geometry').set_crs(
            self.common_metric_crs).to_crs(self.common_not_metric_crs)
        
        blocks['coords'] = blocks.apply(lambda row: tuple([row['geometry'].x,
                                                            row['geometry'].y]), axis=1)
        

        logger.info("Making distance matrix")

        distance_matrix = pd.DataFrame(columns=blocks[block_id_column], 
                                       index=blocks[block_id_column])
        
        block_ids = list(blocks[block_id_column])

        make_calcs = np.vectorize(lambda current_block: Euclidean._calc_dist(
            current_block, distance_matrix, blocks, block_ids, block_id_column))
        

        logger.info("Calculating euclidean distances")
        distance_matrix = make_calcs(block_ids)

        logger.info("Done!")

        return distance_matrix
