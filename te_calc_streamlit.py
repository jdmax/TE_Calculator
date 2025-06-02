import streamlit as st
import pandas as pd
import numpy as np

# Set page config
st.set_page_config(
    page_title="TE Calculator",
    page_icon="🔬",
    layout="wide"
)

# Constants
NUCLEAR_MAGNETON = 5.05e-27  # J/T
BOLTZMANN_CONSTANT = 1.38e-23  # J/K


def calculate_te_polarization(temperature, field, species_magneton, spin):
    """Calculate thermal equilibrium polarization using Boltzmann distribution"""
    magnetic_moment = species_magneton * NUCLEAR_MAGNETON
    x = magnetic_moment * field / (BOLTZMANN_CONSTANT * temperature)

    if spin == 0.5:
        # Spin-1/2 particles (proton, tritium, 13C)
        return np.tanh(x / 2)
    elif spin == 1.0:
        # Spin-1 particles (deuteron)
        # P = 4*tanh(x) / (3 + tanh(x)^2) where x = μB/(kT)
        tanh_x = np.tanh(x)
        return 4 * tanh_x / (3 + tanh_x ** 2)
    else:
        # General case for arbitrary spin J
        # This is more complex and would require the full Brillouin function
        # For now, default to spin-1/2 behavior with a warning
        st.warning(f"Spin {spin} calculation not implemented. Using spin-1/2 approximation.")
        return np.tanh(x / 2)


def calculate_calibration_constant(area, te_polarization):
    """Calculate calibration constant from area and TE polarization"""
    return area / te_polarization


def load_data_from_text(uploaded_file):
    """Load data from uploaded text file"""
    try:
        # Try different delimiters
        content = uploaded_file.read().decode('utf-8')
        lines = content.strip().split('\n')

        # Try to detect delimiter
        first_line = lines[0] if lines else ""
        if '\t' in first_line:
            delimiter = '\t'
        elif ',' in first_line:
            delimiter = ','
        else:
            delimiter = ' '

        # Read data
        data = []
        for line in lines:
            if line.strip():
                parts = [x.strip() for x in line.split(delimiter) if x.strip()]
                if len(parts) >= 2:  # At least temperature and area
                    try:
                        temp = float(parts[0])
                        area = float(parts[1])
                        data.append([temp, area])
                    except ValueError:
                        continue

        if data:
            return pd.DataFrame(data, columns=['Temperature (K)', 'Area'])
        else:
            st.error("Could not parse data from file. Expected format: Temperature Area (tab/comma/space separated)")
            return None

    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None


# Title and description
st.title("🔬 Thermal Equilibrium (TE) Calculator")
st.markdown("Calculate thermal equilibrium polarization and calibration constants for NMR experiments")

# Sidebar for parameters
st.sidebar.header("Experimental Parameters")
field = st.sidebar.number_input("Magnetic Field (T)", value=5.0, min_value=0.1, max_value=20.0, step=0.1)
species = st.sidebar.selectbox("Nuclear Species", ["Proton", "Deuteron", "Tritium", "13C", "Custom"])

# Nuclear species database with spin and magneton values
species_data = {
    "Proton": {"magneton": 2.79268, "spin": 0.5},
    "Deuteron": {"magneton": 0.857438, "spin": 1.0},
    "Tritium": {"magneton": 2.97896, "spin": 0.5},
    "13C": {"magneton": 0.70241, "spin": 0.5}
}

if species in species_data:
    species_magneton = st.sidebar.number_input("Species Magneton",
                                               value=species_data[species]["magneton"],
                                               step=0.001, format="%.5f")
    spin = st.sidebar.number_input("Nuclear Spin",
                                   value=species_data[species]["spin"],
                                   step=0.5, format="%.1f")
    st.sidebar.info(
        f"**{species}**\nSpin: {species_data[species]['spin']}\nMagneton: {species_data[species]['magneton']}")
else:
    species_magneton = st.sidebar.number_input("Species Magneton", value=1.0, step=0.001, format="%.5f")
    spin = st.sidebar.number_input("Nuclear Spin", value=0.5, step=0.5, format="%.1f")

# File upload section
st.header("📁 Data Import")
uploaded_file = st.file_uploader("Upload data file (txt, csv)", type=['txt', 'csv'])

# Initialize session state for data
if 'te_data' not in st.session_state:
    # Default data from your spreadsheet
    default_data = {
        'Temperature (K)': [1.51904183, 1.51912469, 1.518627157, 1.517612724, 1.516694851],
        'Area': [-0.003, -0.001823133297, -0.001748471989, -0.001676877726, -0.001780518129]
    }
    st.session_state.te_data = pd.DataFrame(default_data)

# Handle file upload
if uploaded_file is not None:
    uploaded_data = load_data_from_text(uploaded_file)
    if uploaded_data is not None:
        st.session_state.te_data = uploaded_data
        st.success(f"Successfully loaded {len(uploaded_data)} data points")

# Data input section
st.header("📊 Data Table")
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("Editable Data Table")

    # Add new row button
    if st.button("➕ Add New Row"):
        new_row = pd.DataFrame({'Temperature (K)': [1.5], 'Area': [0.0]})
        st.session_state.te_data = pd.concat([st.session_state.te_data, new_row], ignore_index=True)

    # Display editable table
    edited_data = st.data_editor(
        st.session_state.te_data,
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "Temperature (K)": st.column_config.NumberColumn(
                "Temperature (K)",
                min_value=0.01,
                max_value=10.0,
                step=0.001,
                format="%.6f"
            ),
            "Area": st.column_config.NumberColumn(
                "Area",
                step=0.000001,
                format="%.9f"
            )
        }
    )

    # Update session state
    st.session_state.te_data = edited_data

with col2:
    st.subheader("Quick Actions")

    if st.button("🗑️ Clear All Data"):
        st.session_state.te_data = pd.DataFrame(columns=['Temperature (K)', 'Area'])
        st.rerun()

    if st.button("📄 Load Sample Data"):
        sample_data = {
            'Temperature (K)': [1.51904183, 1.51912469, 1.518627157, 1.517612724, 1.516694851],
            'Area': [-0.003, -0.001823133297, -0.001748471989, -0.001676877726, -0.001780518129]
        }
        st.session_state.te_data = pd.DataFrame(sample_data)
        st.rerun()

# Calculations section
if not st.session_state.te_data.empty and len(st.session_state.te_data) > 0:
    st.header("🧮 Calculations")

    # Perform calculations
    df = st.session_state.te_data.copy()

    # Calculate TE polarization for each point
    df['TE Polarization'] = df['Temperature (K)'].apply(
        lambda t: calculate_te_polarization(t, field, species_magneton, spin)
    )

    # Calculate calibration constant
    df['Calibration Constant'] = df.apply(
        lambda row: calculate_calibration_constant(row['Area'], row['TE Polarization']),
        axis=1
    )

    # Display results table
    st.subheader("📋 Results Table")
    st.dataframe(
        df.style.format({
            'Temperature (K)': '{:.6f}',
            'Area': '{:.9f}',
            'TE Polarization': '{:.9f}',
            'Calibration Constant': '{:.6f}'
        }),
        use_container_width=True
    )

    # Statistics
    st.header("📈 Statistical Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Summary Statistics")

        stats_data = {
            'Parameter': ['Temperature (K)', 'Area', 'TE Polarization', 'Calibration Constant'],
            'Mean': [
                df['Temperature (K)'].mean(),
                df['Area'].mean(),
                df['TE Polarization'].mean(),
                df['Calibration Constant'].mean()
            ],
            'Std Dev': [
                df['Temperature (K)'].std(),
                df['Area'].std(),
                df['TE Polarization'].std(),
                df['Calibration Constant'].std()
            ],
            'Min': [
                df['Temperature (K)'].min(),
                df['Area'].min(),
                df['TE Polarization'].min(),
                df['Calibration Constant'].min()
            ],
            'Max': [
                df['Temperature (K)'].max(),
                df['Area'].max(),
                df['TE Polarization'].max(),
                df['Calibration Constant'].max()
            ]
        }

        stats_df = pd.DataFrame(stats_data)
        st.dataframe(
            stats_df.style.format({
                'Mean': '{:.6f}',
                'Std Dev': '{:.6f}',
                'Min': '{:.6f}',
                'Max': '{:.6f}'
            }),
            use_container_width=True
        )

    with col2:
        st.subheader("🎯 Key Results")

        # Display key metrics
        st.metric("Average Temperature", f"{df['Temperature (K)'].mean():.6f} K",
                  delta=f"±{df['Temperature (K)'].std():.6f}")
        st.metric("Average TE Polarization", f"{df['TE Polarization'].mean():.9f}",
                  delta=f"±{df['TE Polarization'].std():.9f}")
        st.metric("Average Calibration Constant", f"{df['Calibration Constant'].mean():.3f}",
                  delta=f"±{df['Calibration Constant'].std():.3f}")
        st.metric("Number of Points", len(df))

    # Visualizations
    st.header("📊 Visualizations")

    # Create charts with Streamlit's native functions
    col1, col2 = st.columns(2)

    with col1:
        # Temperature histogram using bar chart
        st.subheader("Temperature Distribution")
        temp_counts, temp_bins = np.histogram(df['Temperature (K)'], bins=10)
        temp_chart_data = pd.DataFrame({
            'Temperature Range': [f"{temp_bins[i]:.4f}-{temp_bins[i + 1]:.4f}" for i in range(len(temp_counts))],
            'Count': temp_counts
        })
        st.bar_chart(temp_chart_data.set_index('Temperature Range'))

        # Area vs Calibration Constant scatter
        st.subheader("Area vs Calibration Constant")
        scatter_data = df[['Area', 'Calibration Constant']].copy()
        st.scatter_chart(scatter_data.set_index('Area'))

    with col2:
        # TE Polarization vs Temperature scatter
        st.subheader("TE Polarization vs Temperature")
        te_scatter_data = df[['Temperature (K)', 'TE Polarization']].copy()
        st.scatter_chart(te_scatter_data.set_index('Temperature (K)'))

        # Calibration Constant histogram
        st.subheader("Calibration Constant Distribution")
        cal_counts, cal_bins = np.histogram(df['Calibration Constant'], bins=10)
        cal_chart_data = pd.DataFrame({
            'Calibration Range': [f"{cal_bins[i]:.2f}-{cal_bins[i + 1]:.2f}" for i in range(len(cal_counts))],
            'Count': cal_counts
        })
        st.bar_chart(cal_chart_data.set_index('Calibration Range'))

    # Additional visualizations
    st.subheader("🎨 Advanced Visualizations")

    col1, col2 = st.columns(2)

    with col1:
        # Correlation matrix as a dataframe
        st.subheader("Correlation Matrix")
        correlation_matrix = df[['Temperature (K)', 'Area', 'TE Polarization', 'Calibration Constant']].corr()


        # Display correlation matrix with color formatting
        def color_correlation(val):
            """Color code correlation values"""
            if abs(val) > 0.7:
                return 'background-color: lightcoral' if val < 0 else 'background-color: lightgreen'
            elif abs(val) > 0.4:
                return 'background-color: lightyellow'
            else:
                return 'background-color: lightblue'


        styled_corr = correlation_matrix.style.applymap(color_correlation).format("{:.3f}")
        st.dataframe(styled_corr, use_container_width=True)

        st.caption(
            "🔴 Strong negative correlation | 🟢 Strong positive correlation | 🟡 Moderate correlation | 🔵 Weak correlation")

    with col2:
        # Line chart showing all parameters over measurement index
        st.subheader("Parameter Trends")

        # Normalize data for comparison
        df_normalized = df.copy()
        for col in ['Temperature (K)', 'Area', 'TE Polarization', 'Calibration Constant']:
            mean_val = df_normalized[col].mean()
            std_val = df_normalized[col].std()
            if std_val != 0:
                df_normalized[f'{col} (normalized)'] = (df_normalized[col] - mean_val) / std_val
            else:
                df_normalized[f'{col} (normalized)'] = 0

        # Create line chart with normalized data
        norm_cols = [col for col in df_normalized.columns if '(normalized)' in col]
        trend_data = df_normalized[norm_cols].copy()
        trend_data.index = range(1, len(trend_data) + 1)  # Measurement number

        st.line_chart(trend_data)
        st.caption("All parameters normalized (z-score) for comparison")

    # Summary statistics table
    st.subheader("📈 Data Summary")

    # Create a comprehensive summary
    summary_stats = pd.DataFrame({
        'Parameter': ['Temperature (K)', 'Area', 'TE Polarization', 'Calibration Constant'],
        'Mean': [
            df['Temperature (K)'].mean(),
            df['Area'].mean(),
            df['TE Polarization'].mean(),
            df['Calibration Constant'].mean()
        ],
        'Std Dev': [
            df['Temperature (K)'].std(),
            df['Area'].std(),
            df['TE Polarization'].std(),
            df['Calibration Constant'].std()
        ],
        'Min': [
            df['Temperature (K)'].min(),
            df['Area'].min(),
            df['TE Polarization'].min(),
            df['Calibration Constant'].min()
        ],
        'Max': [
            df['Temperature (K)'].max(),
            df['Area'].max(),
            df['TE Polarization'].max(),
            df['Calibration Constant'].max()
        ],
        'Range': [
            df['Temperature (K)'].max() - df['Temperature (K)'].min(),
            df['Area'].max() - df['Area'].min(),
            df['TE Polarization'].max() - df['TE Polarization'].min(),
            df['Calibration Constant'].max() - df['Calibration Constant'].min()
        ]
    })

    # Format the summary table
    formatted_summary = summary_stats.style.format({
        'Mean': '{:.6f}',
        'Std Dev': '{:.6f}',
        'Min': '{:.6f}',
        'Max': '{:.6f}',
        'Range': '{:.6f}'
    })

    st.dataframe(formatted_summary, use_container_width=True)

    # Individual parameter charts
    if len(df) > 3:
        st.subheader("📊 Individual Parameter Analysis")

        # Allow user to select which parameter to analyze
        param_to_analyze = st.selectbox(
            "Select parameter for detailed analysis:",
            ['Temperature (K)', 'Area', 'TE Polarization', 'Calibration Constant']
        )

        col1, col2 = st.columns(2)

        with col1:
            # Show parameter over measurement index
            param_data = pd.DataFrame({
                'Measurement': range(1, len(df) + 1),
                param_to_analyze: df[param_to_analyze]
            })
            st.line_chart(param_data.set_index('Measurement'))
            st.caption(f"{param_to_analyze} vs Measurement Number")

        with col2:
            # Show simple statistics
            param_values = df[param_to_analyze]
            st.metric("Mean", f"{param_values.mean():.6f}")
            st.metric("Standard Deviation", f"{param_values.std():.6f}")
            st.metric("Coefficient of Variation", f"{(param_values.std() / param_values.mean() * 100):.2f}%")
            if len(param_values) > 1:
                st.metric("Range", f"{param_values.max() - param_values.min():.6f}")

    # Export functionality
    st.header("💾 Export Data")

    col1, col2 = st.columns(2)

    with col1:
        # CSV export
        csv = df.to_csv(index=False)
        st.download_button(
            label="📄 Download as CSV",
            data=csv,
            file_name="te_calculations.csv",
            mime="text/csv"
        )

    with col2:
        # Text export
        text_output = df.to_string(index=False)
        st.download_button(
            label="📝 Download as Text",
            data=text_output,
            file_name="te_calculations.txt",
            mime="text/plain"
        )

else:
    st.info("👆 Add some data points to see calculations and analysis!")

# Footer with constants and formulas
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔬 Physics Constants")
st.sidebar.markdown(f"**Nuclear Magneton:** {NUCLEAR_MAGNETON:.2e} J/T")
st.sidebar.markdown(f"**Boltzmann Constant:** {BOLTZMANN_CONSTANT:.2e} J/K")

st.sidebar.markdown("### 📐 Formulas")
st.sidebar.markdown("**TE Polarization (Spin-1/2):**")
st.sidebar.latex(r"P = \tanh\left(\frac{\mu B}{2 k T}\right)")
st.sidebar.markdown("**TE Polarization (Spin-1):**")
st.sidebar.latex(r"P = \frac{4\tanh(x)}{3 + \tanh^2(x)}")
st.sidebar.markdown("where x = μB/kT")
st.sidebar.markdown("**Calibration Constant:**")
st.sidebar.latex(r"C = \frac{\text{Area}}{P}")

st.sidebar.markdown("---")
st.sidebar.markdown("*TE Calculator, J. Maxwell 2025*")