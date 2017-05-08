import yields
import pytest
import numpy as np

# create simple object for testing
@pytest.fixture
def yields_test_case():
    return yields.Yields("test")

def test_metals_sum(yields_test_case):
    assert yields_test_case._metals_sum() == 55 - 3  # sum(3..10)
    # this is because the test has elements 1 to 10, with an abundance of 1 to
    # 10, and we exclude H and He

def test_normalize_metals(yields_test_case):
    """Test whether or not the normalization is working properly"""
    yields_test_case.normalize_metals(1)
    assert np.isclose(yields_test_case._metals_sum(), 1)
    assert yields_test_case.H_1 == 1.0 / 52.0  # 1 / sum(3..10)
    assert yields_test_case.F_9 == 9.0 / 52.0
    assert yields_test_case.Na_10 == 10.0 / 52.0
    # the total amount must be larger than the amount of metals we normalized 
    # to, since H and He are included. 
    assert sum(yields_test_case._abundances.values()) > 1.0

    # normalize to a value other than 1
    yields_test_case.normalize_metals(25.0)
    assert np.isclose(yields_test_case._metals_sum(), 25.0)
    assert yields_test_case.H_1 == 25.0 / 52.0  # 1 / sum(1..10)
    assert yields_test_case.F_9 == 9.0 * 25.0 / 52.0
    assert np.isclose(yields_test_case.Na_10, 10.0 * 25.0 / 52.0)
    # that one required isclose for whatever reason. 
    assert sum(yields_test_case._abundances.values()) > 25.0

def test_set_metallicity_error_checking(yields_test_case):
    """Metallicities are only vaild between zero and one."""
    with pytest.raises(ValueError):
        yields_test_case.set_metallicity(2)
    with pytest.raises(ValueError):
        yields_test_case.set_metallicity(-0.001)
    with pytest.raises(ValueError):
        yields_test_case.set_metallicity(1.001)

def test_interpolate_z_test(yields_test_case):
    """Test whether the interpolation is working correctly in the test case
    
    In the test case, each isotope goes between the atomic number and the 
    atomic number plus one at metallicities of 0 and 1. 
    So at Z=0, H=1, and at Z=1, H=2"""
    # at zero metallicity, each should be the atomic number
    yields_test_case.set_metallicity(0)
    assert yields_test_case.H_1 == 1.0
    assert yields_test_case.F_9 == 9.0
    assert yields_test_case.Na_10 == 10.0

    # then at all metallicity, each should be the atomic number plus one
    yields_test_case.set_metallicity(1.0)
    assert yields_test_case.H_1 == 2.0
    assert yields_test_case.F_9 == 10.0
    assert yields_test_case.Na_10 == 11.0

    # then test a point halfway in between in log space. This is a little weird,
    # since 0 is at -4 in log space according to my definition. So halfway
    # between -4 and 0=log(1) in log space is log(x) = -2. The resulting
    # value should be atomic number + 0.5
    yields_test_case.set_metallicity(10**-2)
    assert yields_test_case.H_1 == 1.5
    assert yields_test_case.F_9 == 9.5
    assert yields_test_case.Na_10 == 10.5


def test_get_iwamoto_path():
    """Tests the function that gets the path of the Iwamoto yields"""
    file_loc =  yields._get_data_path(yields.iwamoto_file)
    # I know the first line, so I can read that and see what it is
    iwamoto_file = open(file_loc, "r")
    assert iwamoto_file.readline() == "# Table 3 from Iwamoto et al 1999\n"

def test_get_nomoto_path():
    """Tests the function that gets the path of the Iwamoto yields"""
    file_loc =  yields._get_data_path(yields.nomoto_file)
    # I know the first line, so I can read that and see what it is
    iwamoto_file = open(file_loc, "r")
    assert iwamoto_file.readline() == "# Table 3 from Nomoto et al 2006\n"

def test_iwamoto_element_parsing():
    """Tests turning the format of the Iwamoto output into the format this
    class needs"""
    assert yields._parse_iwamoto_element("^{8}O") == "O_8"
    assert yields._parse_iwamoto_element("^{12}C") == "C_12"
    assert yields._parse_iwamoto_element("^{55}Mn") == "Mn_55"
    assert yields._parse_iwamoto_element("^{68}Zn") == "Zn_68"

def test_iwamoto_model_parsing():
    """Tests getting the model itself out of the iwamoto name"""
    assert yields._parse_iwamoto_model("iwamoto_99_Ia_W7") == "W7"
    assert yields._parse_iwamoto_model("iwamoto_99_Ia_W70") == "W70"
    assert yields._parse_iwamoto_model("iwamoto_99_Ia_WDD1") == "WDD1"
    assert yields._parse_iwamoto_model("iwamoto_99_Ia_WDD2") == "WDD2"
    assert yields._parse_iwamoto_model("iwamoto_99_Ia_WDD3") == "WDD3"
    assert yields._parse_iwamoto_model("iwamoto_99_Ia_CDD1") == "CDD1"
    assert yields._parse_iwamoto_model("iwamoto_99_Ia_CDD2") == "CDD2"
    with pytest.raises(ValueError):
        yields._parse_iwamoto_model("iwamsdfs")
    with pytest.raises(ValueError):
        yields._parse_iwamoto_model("iwamoto_99_Ia_wer")  #not a valid model

def test_make_iwamoto_w7():
    iwamoto_test = yields.Yields("iwamoto_99_Ia_W7")
    for z in [0, 0.5, 1]:
        iwamoto_test.set_metallicity(z)
        assert iwamoto_test.C_12 == 4.83E-02
        assert iwamoto_test.Cl_35 == 1.37E-04
        assert iwamoto_test.Zn_68 == 1.74E-08
        assert iwamoto_test.Zn_64 == 1.06E-05
        with pytest.raises(AttributeError):
            iwamoto_test.U_135
        with pytest.raises(AttributeError):
            iwamoto_test.H_1

def test_make_iwamoto_w70():
    iwamoto_test = yields.Yields("iwamoto_99_Ia_W70")
    for z in [0, 0.5, 1]:
        iwamoto_test.set_metallicity(z)
        assert iwamoto_test.C_12 == 5.08E-02
        assert iwamoto_test.Cl_35 == 1.06E-05
        assert iwamoto_test.Zn_68 == 1.13E-08
        assert iwamoto_test.Zn_64 == 7.01E-05
        with pytest.raises(AttributeError):
            iwamoto_test.U_135
        with pytest.raises(AttributeError):
            iwamoto_test.H_1

def test_make_iwamoto_wdd1():
    iwamoto_test = yields.Yields("iwamoto_99_Ia_WDD1")
    for z in [0, 0.5, 1]:
        iwamoto_test.set_metallicity(z)
        assert iwamoto_test.C_12 == 5.42E-03
        assert iwamoto_test.Cl_35 == 9.28E-05
        assert iwamoto_test.Zn_68 == 7.44E-08
        assert iwamoto_test.Zn_64 == 3.71E-06
        with pytest.raises(AttributeError):
            iwamoto_test.U_135
        with pytest.raises(AttributeError):
            iwamoto_test.H_1

def test_make_iwamoto_wdd2():
    iwamoto_test = yields.Yields("iwamoto_99_Ia_WDD2")
    for z in [0, 0.5, 1]:
        iwamoto_test.set_metallicity(z)
        assert iwamoto_test.C_12 == 8.99E-03
        assert iwamoto_test.Cl_35 == 7.07E-05
        assert iwamoto_test.Zn_68 == 8.81E-08
        assert iwamoto_test.Zn_64 == 3.10E-05
        with pytest.raises(AttributeError):
            iwamoto_test.U_135
        with pytest.raises(AttributeError):
            iwamoto_test.H_1

def test_make_iwamoto_wdd3():
    iwamoto_test = yields.Yields("iwamoto_99_Ia_WDD3")
    for z in [0, 0.5, 1]:
        iwamoto_test.set_metallicity(z)
        assert iwamoto_test.C_12 == 1.66E-02
        assert iwamoto_test.Cl_35 == 5.33E-05
        assert iwamoto_test.Zn_68 == 9.42E-08
        assert iwamoto_test.Zn_64 == 5.76E-05
        with pytest.raises(AttributeError):
            iwamoto_test.U_135
        with pytest.raises(AttributeError):
            iwamoto_test.H_1

def test_make_iwamoto_cdd1():
    iwamoto_test = yields.Yields("iwamoto_99_Ia_CDD1")
    for z in [0, 0.5, 1]:
        iwamoto_test.set_metallicity(z)
        assert iwamoto_test.C_12 == 9.93E-03
        assert iwamoto_test.Cl_35 == 9.03E-05
        assert iwamoto_test.Zn_68 == 3.08E-09
        assert iwamoto_test.Zn_64 == 1.87E-06
        with pytest.raises(AttributeError):
            iwamoto_test.U_135
        with pytest.raises(AttributeError):
            iwamoto_test.H_1

def test_make_iwamoto_cdd2():
    iwamoto_test = yields.Yields("iwamoto_99_Ia_CDD2")
    for z in [0, 0.5, 1]:
        iwamoto_test.set_metallicity(z)
        assert iwamoto_test.C_12 == 5.08E-03
        assert iwamoto_test.Cl_35 == 6.56E-05
        assert iwamoto_test.Zn_68 == 3.03E-08
        assert iwamoto_test.Zn_64 == 3.96E-05
        with pytest.raises(AttributeError):
            iwamoto_test.U_135
        with pytest.raises(AttributeError):
            iwamoto_test.H_1

def test_met_log():
    """Tests the metallicity log function. Is just like log, but returns a
    fixed value for 0."""
    assert yields._metallicity_log(0) == -4
    assert yields._metallicity_log(1) == 0
    assert yields._metallicity_log(0.01) == -2
    assert yields._metallicity_log(100) == 2

def test_normalization_stability(yields_test_case):
    """Once we set the normalization, the total amount of metals should not
    change. Make sure that is the case. """
    yields_test_case.set_metallicity(0)
    total_metals = 10
    yields_test_case.normalize_metals(total_metals)
    assert np.isclose(yields_test_case._metals_sum(), total_metals)
    # then change the metallicity
    yields_test_case.set_metallicity(0.2)
    assert np.isclose(yields_test_case._metals_sum(), total_metals)
    # then do it again
    yields_test_case.set_metallicity(1)
    assert np.isclose(yields_test_case._metals_sum(), total_metals)

def test_nomoto_parser():
    """Test the funciton that takes the name and element from the Nomoto file
    and puts it in the right format that we want."""
    assert yields._parse_nomoto_element("01", "p") == "H_1"
    assert yields._parse_nomoto_element("02", "d") == "H_2"
    assert yields._parse_nomoto_element("09", "Be") == "Be_9"
    assert yields._parse_nomoto_element("24", "Na") == "Na_24"
    assert yields._parse_nomoto_element("30", "Si") == "Si_30"


# create simple object for testing
@pytest.fixture
def yields_nomoto():
    return yields.Yields("nomoto_06_II")

def test_make_nomoto_at_zero_met(yields_nomoto):
    """Test that the Nomoto yields return the correct values. 
    
    I will test at all the given metallicities in the paper, to make sure it 
    works at the values in the table. I will put all the metallicity points
    in the same function, to make sure that is working correctly too."""
    yields_nomoto.set_metallicity(0)
    assert yields_nomoto.H_1 == 3.28E-02
    assert yields_nomoto.H_2 == 5.76E-18
    assert yields_nomoto.Si_28 == 8.11E-04
    assert yields_nomoto.Ca_48 == 8.04E-15
    assert yields_nomoto.Zn_64 == 6.32E-07
    assert yields_nomoto.Ge_74 == 1.33E-14
    with pytest.raises(AttributeError):
        yields_nomoto.U_135

    yields_nomoto.set_metallicity(0.001)
    assert yields_nomoto.H_1 == 3.14E-02
    assert yields_nomoto.H_2 == 2.21E-15
    assert yields_nomoto.Si_28 == 7.09E-04
    assert yields_nomoto.Ca_48 == 6.91E-10
    assert yields_nomoto.Zn_64 == 5.74E-07
    assert yields_nomoto.Ge_74 == 2.18E-08
    with pytest.raises(AttributeError):
        yields_nomoto.U_135

    yields_nomoto.set_metallicity(0.004)
    assert yields_nomoto.H_1 == 2.96E-02
    assert yields_nomoto.H_2 == 1.97E-16
    assert yields_nomoto.Si_28 == 6.17E-04
    assert yields_nomoto.Ca_48 == 2.93E-09
    assert yields_nomoto.Zn_64 == 5.07E-07
    assert yields_nomoto.Ge_74 == 1.35E-07
    with pytest.raises(AttributeError):
        yields_nomoto.U_135

    yields_nomoto.set_metallicity(0.02)
    assert yields_nomoto.H_1 == 2.45E-02
    assert yields_nomoto.H_2 == 5.34E-16
    assert yields_nomoto.Si_28 == 4.55E-04
    assert yields_nomoto.Ca_48 == 1.07E-08
    assert yields_nomoto.Zn_64 == 4.43E-07
    assert yields_nomoto.Ge_74 == 7.93E-07
    with pytest.raises(AttributeError):
        yields_nomoto.U_135

def test_make_nomoto_interpolation_range(yields_nomoto):
    """Tests that the interpolation is returning values in the range we need."""
    # first just test that the values are in the right range (ie between the
    # abundances of the metallicities that span the metallicity used.
    yields_nomoto.set_metallicity(0.002)
    assert 3.14E-2 > yields_nomoto.H_1 > 2.96E-2
    assert 7.09E-4 > yields_nomoto.Si_28 > 6.17E-4
    assert 3.34E-5 > yields_nomoto.Ca_40 > 3.02E-5

    # try a different metallicity value
    yields_nomoto.set_metallicity(0.01)
    assert 2.96E-2 > yields_nomoto.H_1 > 2.45E-2
    assert 6.17E-4 > yields_nomoto.Si_28 > 4.55E-4
    assert 3.02E-5 > yields_nomoto.Ca_40 > 2.39E-5

def test_make_nomoto_interpolation_values(yields_nomoto):
    """Tests that the interpolation is working correctly by directly testing 
       values, not just checking their range."""
    # I want to get a metallicity directly in between in log space, which can
    # be gotten using the logspace function
    middle = np.logspace(yields._metallicity_log(0),
                         yields._metallicity_log(0.001), 3)[1]  # get middle val
    yields_nomoto.set_metallicity(middle)
    assert np.isclose(yields_nomoto.H_1, np.mean([3.28E-2, 3.14E-2]))
    assert np.isclose(yields_nomoto.Ca_46, np.mean([5.69E-14, 2.06E-10]))
    assert np.isclose(yields_nomoto.Ge_74, np.mean([1.33E-14, 2.18E-8]))

    # then repeat for a different metallicity
    middle = np.logspace(yields._metallicity_log(0.004),
                         yields._metallicity_log(0.02), 3)[1]  # get middle val
    yields_nomoto.set_metallicity(middle)
    assert np.isclose(yields_nomoto.H_1, np.mean([2.96E-2, 2.45E-2]))
    assert np.isclose(yields_nomoto.Ca_46, np.mean([8.71E-10, 3.60E-9]))
    assert np.isclose(yields_nomoto.Ge_74, np.mean([1.35E-7, 7.93E-7]))

def test_metallicity_outside_range_nomoto(yields_nomoto):
    """Tests what happens when the metallicity is outside the range the 
    models span. I will assert that it should be the same as the yields of the
    model that is at the extreme. """
    yields_nomoto.set_metallicity(0.99)
    assert yields_nomoto.H_1 == 2.45E-2
    assert yields_nomoto.H_2 == 5.34E-16
    assert yields_nomoto.O_16 == 6.14E-3
    assert yields_nomoto.Al_27 == 6.53E-5
    assert yields_nomoto.Fe_58 == 2.15E-6
    assert yields_nomoto.Fe_54 == 1.13E-5