import pandas as pd
from scipy import signal
import numpy as np
import io

class StreamlitRamanSpectrumProcessor:
    def __init__(self, laser_wavelength=532):
        """
        Initialize Streamlit Raman spectrum processor
        
        Args:
            laser_wavelength: laser wavelength (nm)
        """
        # 既定ピクセル位置（変更なし）
        self.default_pixel_indexs = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1400, 1800]
        self.pixel_indexs = self.default_pixel_indexs

        # 内部保持用の実体
        self._laser_wavelength = None
        self.default_ethanol_peaks = []  # プロパティ設定時に更新

        # ← ここでプロパティsetterを通して設定（既定ピークを自動反映）
        self.laser_wavelength = laser_wavelength

    # --- レーザー波長プロパティ（設定時に既定ピークを自動切替） ---
    @property
    def laser_wavelength(self):
        return self._laser_wavelength

    @laser_wavelength.setter
    def laser_wavelength(self, value):
        self._laser_wavelength = float(value)
        self._set_default_peaks_by_laser()

    def _set_default_peaks_by_laser(self):
        """レーザー波長に応じて既定のエタノールピーク(cm^-1)を切り替える。"""
        lw = int(round(self._laser_wavelength)) if self._laser_wavelength is not None else None

        if lw == 532:
            # 532nm のとき
            self.default_ethanol_peaks = [5150, 2975, 2930, 2883, 1455, 1280, 1099, 1054, 883, 436, 0]
        elif lw == 785:
            # 785nm のとき
            self.default_ethanol_peaks = [3400, 2975, 2930, 2883, 1455, 1280, 1099, 1054, 883, 436, 0]
        else:
            # その他の波長は従来の既定値にフォールバック（必要なら調整してください）
            self.default_ethanol_peaks = [2975, 2930, 2883, 1455, 1280, 1099, 1054, 883, 436, 0]

    # --- 単位変換 ---
    def wavenumber_to_wavelength(self, wavenumber):
        """Convert wavenumber to wavelength (nm)"""
        return 10000000 / (10000000/self.laser_wavelength - wavenumber)
    
    def wavelength_to_wavenumber(self, wavelength):
        """Convert wavelength (nm) to wavenumber (cm^-1)"""
        result = 10000000/self.laser_wavelength - 10000000/wavelength
        if np.isscalar(wavelength):
            return int(result)
        else:
            return result.astype(int)

    # --- CSV 読み込み ---
    def read_csv_data(self, csv_file):
        """Read CSV file data with robust format handling"""
        try:
            # First, read the file content as text to handle various formats
            content = csv_file.getvalue()
            if isinstance(content, bytes):
                content = content.decode('utf-8')
            
            lines = content.strip().split('\n')
            
            # Remove comment lines and empty lines
            data_lines = []
            header_line = None
            
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                if line.startswith('#'):  # Skip comment lines
                    continue
                
                # Check if this might be a header line (contains non-numeric data)
                if header_line is None and any(not part.replace('.', '').replace('-', '').replace('+', '').replace('e', '').replace('E', '').isdigit() 
                                            for part in line.split(',') if part.strip()):
                    header_line = line
                else:
                    data_lines.append(line)
            
            # If no clear header found, create one based on number of columns
            if header_line is None and data_lines:
                first_data_line = data_lines[0].split(',')
                num_cols = len(first_data_line)
                if num_cols >= 2:
                    header_parts = ['WaveNumber']
                    if num_cols >= 3:
                        header_parts.append('DarkSpectrum')
                        for i in range(2, num_cols):
                            header_parts.append(f'Spectrum{i-1}')
                    else:
                        header_parts.append('Spectrum1')
                    header_line = ','.join(header_parts)
            
            # Combine header and data
            if header_line:
                csv_content = header_line + '\n' + '\n'.join(data_lines)
            else:
                csv_content = '\n'.join(data_lines)
            
            # Read with pandas
            df = pd.read_csv(io.StringIO(csv_content))
            
            # Clean up the dataframe - remove any completely empty rows/columns
            df = df.dropna(how='all')  # Remove rows where all values are NaN
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]  # Remove unnamed columns
            
            # Handle the CSV structure
            columns = df.columns.tolist()
            
            # Format 1: Has PixelIndex column
            if 'PixelIndex' in columns or any('pixel' in col.lower() for col in columns):
                pixel_col = next((col for col in columns if 'pixel' in col.lower()), columns[0])
                pixel_index = df[pixel_col].values
                
                # Find spectrum data columns (skip pixel, wavenumber, dark spectrum)
                skip_cols = [pixel_col]
                if 'WaveNumber' in columns:
                    skip_cols.append('WaveNumber')
                if 'DarkSpectrum' in columns:
                    skip_cols.append('DarkSpectrum')
                
                spectrum_cols = [col for col in columns if col not in skip_cols]
                if len(spectrum_cols) > 0:
                    spectrum_data = df[spectrum_cols].mean(axis=1).values
                else:
                    raise ValueError("No spectrum data columns found!")
                    
            # Format 2: First column is WaveNumber, followed by spectrum data
            elif 'WaveNumber' in columns or len(columns) >= 2:
                # Generate pixel index starting from 0
                pixel_index = np.arange(len(df))
                
                # Find spectrum data columns (skip wavenumber and dark spectrum)
                skip_cols = []
                if 'WaveNumber' in columns:
                    skip_cols.append('WaveNumber')
                if 'DarkSpectrum' in columns:
                    skip_cols.append('DarkSpectrum')
                
                spectrum_cols = [col for col in columns if col not in skip_cols]
                if len(spectrum_cols) > 0:
                    # Convert to numeric, replacing any non-numeric values with NaN
                    for col in spectrum_cols:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # Calculate mean, ignoring NaN values
                    spectrum_data = df[spectrum_cols].mean(axis=1, skipna=True).values
                    
                    # Remove any NaN values
                    valid_indices = ~np.isnan(spectrum_data)
                    pixel_index = pixel_index[valid_indices]
                    spectrum_data = spectrum_data[valid_indices]
                else:
                    return None, None
                    
            # Format 3: Simple format with just numerical data
            else:
                # Assume first column is wavenumber/wavelength, rest are spectrum data
                pixel_index = np.arange(len(df))
                spectrum_cols = df.columns[1:]  # Skip first column
                if len(spectrum_cols) > 0:
                    # Convert to numeric
                    for col in spectrum_cols:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    spectrum_data = df[spectrum_cols].mean(axis=1, skipna=True).values
                else:
                    # Only one column, treat it as spectrum data
                    df.iloc[:, 0] = pd.to_numeric(df.iloc[:, 0], errors='coerce')
                    spectrum_data = df.iloc[:, 0].values
                    spectrum_cols = [df.columns[0]]
            
            # Final validation
            if len(pixel_index) == 0 or len(spectrum_data) == 0:
                raise ValueError("No valid pixel index or spectrum data found in the file.")
            
            self.pixel_indexs = pixel_index
            # Display file format information
            return pixel_index, spectrum_data
            
        except Exception as e:
            # Show file preview to help debug
            raise ValueError(f"Error reading CSV file: {e}")
    
    def find_peaks_in_spectrum(self, pixel_index, spectrum_data, prominence=50, distance=20):
        """Find peaks in spectrum"""
        peaks, properties = signal.find_peaks(spectrum_data, 
                                            prominence=prominence, 
                                            distance=distance)
        peak_pixels = pixel_index[peaks]
        peak_intensities = spectrum_data[peaks]
        sorted_indices = np.argsort(peak_intensities)[::-1]
        return peak_pixels[sorted_indices], peak_intensities[sorted_indices]
    
    def polynomial_fitting(self, pixel_numbers, wavelengths, max_degree=5):
        """Polynomial fitting"""
        fitting_results = {}
        
        for degree in range(1, max_degree + 1):
            if len(pixel_numbers) > degree:
                coeffs = np.polyfit(pixel_numbers, wavelengths, degree)
                fitted_wavelengths = np.polyval(coeffs, pixel_numbers)
                rmse = np.sqrt(np.mean((wavelengths - fitted_wavelengths)**2))
                r_squared = 1 - np.sum((wavelengths - fitted_wavelengths)**2) / np.sum((wavelengths - np.mean(wavelengths))**2)
                
                fitting_results[degree] = {
                    'coeffs': coeffs,
                    'rmse': rmse,
                    'r_squared': r_squared
                }
        
        if fitting_results:
            best_degree = max(fitting_results.keys(), key=lambda k: fitting_results[k]['r_squared'])
            best_coeffs = fitting_results[best_degree]['coeffs']
            return best_coeffs, best_degree, fitting_results
        return None, None, {}