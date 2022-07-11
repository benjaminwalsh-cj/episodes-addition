import typing
import logging
import json
import pandas as pd
import helpers


logger = logging.basicConfig(level='INFO',
                             format='[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
                             datefmt='%H:%M:%S')


def load_snowflake_config(
        path_to_env_file: str,
        warehouse: str) -> dict:
    '''Load a snowflake configuration file
    Args:
        path_to_env_file (`str`): path to the snowflake configuration json file
        warehouse (`str`): warehouse to use in the connection
    Returns:
        dictionary of configurations
    '''

    try:
        # Get environment variables
        with open(path_to_env_file, 'r') as a:
            environment_variables = json.load(a)
        environment_variables['WAREHOUSE'] = warehouse
    except FileNotFoundError as e:
        logger.error('Provided path is incorrect.')
        raise e

    return environment_variables


def generate_sf_connections(
    env_variables: dict
):
    '''Use helpers.get_sf_connection and the provided warehouse to
    generate an snowflake connection to use in querying
    Args:
        env_variables (`dict`): dictionary of environment variables. Should be
        generated using the `load_snowflake_config` function.
    Returns:
        A snowflake connection
    '''

    try:
        sf_connection = helpers.get_snowflake_connection(
            username=env_variables['SNOWFLAKE_USER'],
            password=env_variables['SNOWFLAKE_PASS'],
            account=env_variables['SNOWFLAKE_ACCOUNT_NCI'],
            warehouse=env_variables['WAREHOUSE'],
            database=None
        )
    except AttributeError as e:
        logger.error('Provided dictionary does not contain all required vars.')
        raise e

    return sf_connection


def generate_episodes_query(
        query: str,
        npis: list[int],
        max_num_exp: int = 10000) -> str:
    '''Iterate through the NPIs to add them to the base query to generate a
    full query that can be executed
    Args:
        base_query (`str`): base query to use
        npis (`list[int]`): list of NPIs to query
        max_num_exp (`int`): max number of expressions to include (< sf limit)
    Returns:
        Complete query string
    '''

    # Get all episodes for the providers if no query is provided
    if query is None:
        query = '''WITH xwalk AS (
    SELECT DISTINCT episode_name, episode_desc 
    FROM pantry_nci.sndbx.egm_all_codes
    )
    SELECT 
        egmNpi.EPISODE_DESCRIPTION
        , xwalk.EPISODE_DESC
        , egmNpi.FK_PROVIDER_ID AS NPI
        , egmNpi.EPISODE_TYPE_CD
        , 1 AS count
    FROM "PROD_CJVRDC"."VRDC"."EGM_NPI" egmNpi
        JOIN xwalk
            ON xwalk.episode_name = egmNpi.EPISODE_DESCRIPTION
    WHERE ( '''

    # Add NPIs to the query
    for i in range(len(npis) // npis + 1):
        if i <= len(npis) // max_num_exp - 1:
            query += f'egmNpi.FK_PROVIDER_ID IN {tuple(npis[i * max_num_exp : (i + 1) * max_num_exp])} OR '
        else:
            query += f'egmNpi.FK_PROVIDER_ID IN {tuple(npis[i * max_num_exp :])})'

    return query


def generate_episodes_dataframe(
        sf_connection,
        query: str = None) -> pd.DataFrame:
    '''Query snowflake to generate a dataframe of episodes that
    a provider has billed
    Args:
        sf_connection (`snowflake_connection`): snowflake connection to use
        the query
        query (`str`): the complete query to execute. Should be generated using
        the `generate_episodes_query` function.
    Return:
        Dataframe of episodes billed by the provided NPIs
    '''
    # Execute query
    df_episodes = helpers.fetch_sf_data(sf_connection, query)

    # Modify NPI to be an integer
    df_episodes['npi'] = df_episodes['npi'].astype(int)

    return df_episodes


def generate_dummy_df_episodes(
        df_episodes: pd.DataFrame,
        prepend_value: str = None) -> pd.DataFrame:
    '''Pivot the episodes dataframe, summing the episodes so each NPI has a
    single line
    Args:
        df_episodes (`pd.DataFrame`): dataframe of episodes billed by NPIs
        prepend_value (`str`): value to prepend to each of the episode columns
    Returns:
        Pivoted pandas dataframe
    '''

    # Pivot, summing the count column

    try:
        df_episodes_dummy = df_episodes.pivot_table(
            values='count',
            index='npi',
            columns='episode_desc',
            aggfunc='sum'
        )
    except AttributeError as e:
        logger.error('Column `count` may be missing from episodes dataframe.')
        raise e

    if prepend_value is not None:
        df_episodes_dummy.set_axis(
            ['epi_' + col for col in df_episodes_dummy.columns],
            axis=1,
            inplace=True
        )

    return df_episodes_dummy
