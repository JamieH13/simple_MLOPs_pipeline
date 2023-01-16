from methods import *

def test_one_hot_encoder():
    test = pd.DataFrame(['Widow', 'Separated', 'Married', 'Married'], columns=['NAME_FAMILY_STATUS'])

    result = OneHotEncode(cols=['NAME_FAMILY_STATUS']).fit_transform(test)

    expected = pd.DataFrame([[0,0,1], [0,1,0], [1,0,0], [1,0,0]], dtype='uint8',columns=['NAME_FAMILY_STATUS_Married',
                                                                                         'NAME_FAMILY_STATUS_Separated',
                                                                                         'NAME_FAMILY_STATUS_Widow'])

    assert result.equals(expected)
