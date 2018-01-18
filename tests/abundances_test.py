from yields import abundances

import pytest
from pytest import approx
import numpy as np

# setup
solar_z = abundances.create_solar_metal_fractions()[0]

abundance_obj = abundances.Abundances()

# -----------------------------------------------------------

#  Test error checking

# -----------------------------------------------------------
abundance_functions_basic = [abundance_obj.z_on_h,
                             abundance_obj.log_z_over_z_sun]
abundance_functions_elts = [abundance_obj.x_on_fe,
                            abundance_obj.x_on_h]


@pytest.mark.parametrize("func", abundance_functions_basic)
def test_init_error_checking_lists_no_elt(func):
    """Lists have to be the same length."""
    with pytest.raises(ValueError):
        func([0.1, 0.2], [0.1, 0.2, 0.3])


@pytest.mark.parametrize("func", abundance_functions_elts)
def test_init_error_checking_lists_elt(func):
    """Lists have to be the same length."""
    with pytest.raises(ValueError):
        func("O", [0.1, 0.2], [0.1, 0.2, 0.3])


@pytest.mark.parametrize("func", abundance_functions_basic)
def test_init_error_checking_z_bounds_no_elt(func):
    """Metallicities must be between zero and one."""
    with pytest.raises(ValueError):
        func(-0.1, 0.1)
    with pytest.raises(ValueError):
        func(0.1, -0.1)
    with pytest.raises(ValueError):
        func(1.1, 0.1)
    with pytest.raises(ValueError):
        func(0.1, 1.1)
    # the sum of the metallicity can't be larger than one, either.
    with pytest.raises(ValueError):
        func(0.6, 0.6)


@pytest.mark.parametrize("func", abundance_functions_elts)
def test_init_error_checking_z_bounds_elt(func):
    """Metallicities must be between zero and one."""
    with pytest.raises(ValueError):
        func("O", -0.1, 0.1)
    with pytest.raises(ValueError):
        func("O", 0.1, -0.1)
    with pytest.raises(ValueError):
        func("O", 1.1, 0.1)
    with pytest.raises(ValueError):
        func("O", 0.1, 1.1)
    # the sum of the metallicity can't be larger than one, either.
    with pytest.raises(ValueError):
        func("O", 0.6, 0.6)


def test_solar_fractions():
    """Checks whether the solar abundances are reasonable. """
    abund = abundances.Abundances()  # values irrelevant
    assert 0.015 < abund.Z_sun < 0.02

    metal_fracs_sum = np.sum(abund.solar_metal_fractions.values())
    metal_fracs_sum -= abund.solar_metal_fractions["H"]
    metal_fracs_sum -= abund.solar_metal_fractions["He"]
    assert np.isclose(metal_fracs_sum, 1)


@pytest.mark.parametrize("func", abundance_functions_basic)
def test_rtype_single(func):
    assert type(func(0.1, 0.2)) == float


@pytest.mark.parametrize("func", abundance_functions_basic)
def test_rtype_single_list(func):
    assert type(func([0.1], [0.2])) == float


@pytest.mark.parametrize("func", abundance_functions_basic)
def test_rtype_multi_list(func):
    assert type(func([0.1, 0.2], [0.2, 0.1])) == np.ndarray


@pytest.mark.parametrize("func", abundance_functions_elts)
def test_rtype_single_elt(func):
    assert type(func("O", 0.1, 0.2)) == float


@pytest.mark.parametrize("func", abundance_functions_elts)
def test_rtype_single_list_elt(func):
    assert type(func("O", [0.1], [0.2])) == float


@pytest.mark.parametrize("func", abundance_functions_elts)
def test_rtype_multi_list_elt(func):
    assert type(func("O", [0.1, 0.2], [0.2, 0.1])) == np.ndarray


# -----------------------------------------------------------

#  Test [Z/H]

# -----------------------------------------------------------
def test_z_on_h_calculation_single_zero():
    """For z zero metallicity object we should have negative infinity. """
    assert np.isneginf(abundance_obj.z_on_h(0, 0))


def test_z_on_h_calculation_single_one_Ia():
    """For an object of metallicity one, we will get an infinite value, since
    we will be dividing by zero. """
    assert np.isposinf(abundance_obj.z_on_h(1, 0))


def test_z_on_h_calculation_single_one_II():
    """For an object of metallicity one, we will get an infinite value, since
    we will be dividing by zero. """
    assert np.isposinf(abundance_obj.z_on_h(0, 1))


def test_z_on_h_calculation_single_solar_Ia():
    """For a solar metallicity object we should get zero. """
    assert abundance_obj.z_on_h(solar_z, 0) == approx(0)


def test_z_on_h_calculation_single_solar_II():
    """For a solar metallicity object we should get zero. """
    assert abundance_obj.z_on_h(0, solar_z) == approx(0)


def test_z_on_h_array():
    """Same as tests above, but at once, to test arrays."""
    test = abundance_obj.z_on_h([solar_z, 0], [0, solar_z])
    real = [0, 0]
    assert test == approx(real)


# -----------------------------------------------------------

#  Test [Fe/H]

# -----------------------------------------------------------
def test_fe_on_h_calculation_single_zero_():
    """For z zero metallicity object we should have negative infinity. """
    assert np.isneginf(abundance_obj.x_on_h("Fe", 0, 0))


def test_fe_on_h_calculation_single_one_Ia():
    """For an object of metallicity one, we will get an infinite value, since
    we will be dividing by zero. """
    assert np.isposinf(abundance_obj.x_on_h("Fe", 1, 0))


def test_fe_on_h_calculation_single_one_II():
    """For an object of metallicity one, we will get an infinite value, since
    we will be dividing by zero. """
    assert np.isposinf(abundance_obj.x_on_h("Fe", 0, 1))


def test_fe_on_h_calculation_single_solar_Ia():
    """For a solar metallicity object we have to manually calculate it using
     the actual yield object. """
    assert abundance_obj.x_on_h("Fe", solar_z, 0) == approx(0.858336)


def test_fe_on_h_calculation_single_solar_II():
    """For a solar metallicity object we have to manually calculate it using
     the actual yield object. """
    assert abundance_obj.x_on_h("Fe", 0, solar_z) == approx(-0.2740616354)


def test_fe_on_h_array():
    test = abundance_obj.x_on_h("Fe", [solar_z, 0], [0, solar_z])
    real = [0.858336, -0.2740616354]
    assert test == approx(real)


# -----------------------------------------------------------

#  Test [X/H] for different elements

# -----------------------------------------------------------
def test_na_on_h_calculation_single_zero():
    """For z zero metallicity object we should have negative infinity. """
    assert np.isneginf(abundance_obj.x_on_h("Na", 0, 0))


def test_na_on_h_calculation_single_one_Ia():
    """For an object of metallicity one, we will get an infinite value, since
    we will be dividing by zero. """
    assert np.isposinf(abundance_obj.x_on_h("Na", 1, 0))


def test_na_on_h_calculation_single_one_II():
    """For an object of metallicity one, we will get an infinite value, since
    we will be dividing by zero. """
    assert np.isposinf(abundance_obj.x_on_h("Na", 0, 1))


def test_na_on_h_calculation_single_solar_Ia():
    """For a solar metallicity object we have to manually calculate it using
     the actual yield object. """
    assert abundance_obj.x_on_h("Na", solar_z, 0) == approx(-1.66083, abs=1E-5)


def test_na_on_h_calculation_single_solar_II():
    """For a solar metallicity object we have to manually calculate it using
     the actual yield object. """
    assert abundance_obj.x_on_h("Na", 0, solar_z) == approx(0.3078309)


def test_na_on_h_array():
    test = abundance_obj.x_on_h("Na", [solar_z, 0], [0, solar_z])
    real = [-1.66083, 0.3078309]
    assert test == approx(real, abs=1E-5)


# -----------------------------------------------------------

#  Test [Fe/Fe]. Should give zero.

# -----------------------------------------------------------
def test_fe_on_fe_calculation_single_zero():
    """For z zero metallicity object we should get some kind of error.
    We are taking a log of 0/0, which apparently give a nan. """
    assert np.isnan(abundance_obj.x_on_fe("Fe", 0, 0))


def test_fe_on_fe_calculation_single_one_Ia():
    """For an object of metallicity one, we will get zero"""
    assert abundance_obj.x_on_fe("Fe", 1, 0) == approx(0)


def test_fe_on_fe_calculation_single_one_II():
    """For an object of metallicity one, we will get zero """
    assert abundance_obj.x_on_fe("Fe", 0, 1) == approx(0)


def test_fe_on_fe_calculation_single_solar_Ia():
    """For a solar metallicity object we get zero.  """
    assert abundance_obj.x_on_fe("Fe", solar_z, 0) == approx(0)


def test_fe_on_fe_calculation_single_solar_II():
    """For a solar metallicity object we get zero.  """
    assert abundance_obj.x_on_fe("Fe", 0, solar_z) == approx(0)


def test_fe_on_fe_array():
    test = abundance_obj.x_on_fe("Fe",
                                 [1, 0, solar_z, 0],
                                 [0, 1, 0, solar_z])
    real = [0, 0, 0, 0]
    assert test == approx(real)


# -----------------------------------------------------------

#  Test [O/Fe]. Should not give zero. I'll calculate the values. I don't need
# to check the metallicity of one points anymore, since there isn't a 1-Z
# anywhere in this code. The solar metallicity is fine.

# -----------------------------------------------------------
def test_o_on_fe_calculation_single_zero():
    """For z zero metallicity object we should get some kind of infinity.
    We are taking a log of 0/0, so who knows what that will give. """
    assert np.isnan(abundance_obj.x_on_fe("O", 0, 0))


def test_o_on_fe_calculation_single_solar_Ia():
    """Calculated by hand.  """
    assert abundance_obj.x_on_fe("O", solar_z, 0) == approx(-1.507779731)


def test_o_on_fe_calculation_single_solar_II():
    """Calculated by hand.  """
    assert abundance_obj.x_on_fe("O", 0, solar_z) == approx(0.3533312216)


def test_o_on_fe_array():
    test = abundance_obj.x_on_fe("O", [solar_z, 0], [0, solar_z])
    real = [-1.507779731, 0.3533312216]
    assert test == approx(real)


# -----------------------------------------------------------

#  Test simple metallicity values

# -----------------------------------------------------------
def test_log_z_z_sun_single_zero():
    """For a zero metallicity object we should get a negative infinity."""
    assert np.isneginf(abundance_obj.log_z_over_z_sun(0, 0))


def test_log_z_z_sun_single_one_Ia():
    """For solar we should get zero."""
    assert abundance_obj.log_z_over_z_sun(1, 0) == approx(1.789274018)


def test_log_z_z_sun_single_one_II():
    """For solar we should get zero."""
    assert abundance_obj.log_z_over_z_sun(0, 1) == approx(1.789274018)


def test_log_z_z_sun_single_solar_Ia():
    """For solar we should get the log of the ratio of 1 and z_sun.
    This was calculated by hand."""
    assert abundance_obj.log_z_over_z_sun(solar_z, 0) == approx(0)


def test_log_z_z_sun_single_solar_II():
    """For solar we should get the log of the ratio of 1 and z_sun.
    This was calculated by hand."""
    assert abundance_obj.log_z_over_z_sun(0, solar_z) == approx(0)


def test_log_z_z_sun_array():
    test = abundance_obj.log_z_over_z_sun([1, 0, solar_z, 0],
                                          [0, 1, 0, solar_z])
    real = [1.789274018, 1.789274018, 0, 0]
    assert test == approx(real)
