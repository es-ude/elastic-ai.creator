----------------------------------------------------------------------------------
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

library work;
use work.userlogicinterface.all;

entity InterfaceStateMachine is
	generic
	(
		control_region		: unsigned(15 downto 0) := x"00ff"
	);
	port(
		clk					: in std_logic;							-- clock
		reset				: in std_logic;							-- reset everything
		
		-- icap interface
		icap_address		: out uint32_t_interface;
		
		-- sram interface
		sram_address		: in std_logic_vector(15 downto 0);
		sram_data_out		: out std_logic_vector(7 downto 0);
		sram_data_in		: in std_logic_vector(7 downto 0);
		sram_rd				: in std_logic;
		sram_wr				: in std_logic;
		
		-- userlogic interface
		userlogic_reset		: out std_logic;
		userlogic_data_in	: out std_logic_vector(7 downto 0);
		userlogic_address	: out std_logic_vector(15 downto 0);
		userlogic_data_out	: in std_logic_vector(7 downto 0);
		userlogic_rd		: out std_logic;
		userlogic_wr		: out std_logic;
		
		leds 				: out std_logic_vector(3 downto 0)
	);
end InterfaceStateMachine;

architecture rtl of InterfaceStateMachine is 
	constant MULTIBOOT : uint16_t := x"0005";
	constant LED : uint16_t := x"0003";
	constant USERLOGIC_CONTROL : uint16_t := x"0004";
	
	signal led_signal : std_logic_vector(3 downto 0) := (others => '0');
	signal userlogic_reset_signal : std_logic := '0';

	signal middleware_data_out : std_logic_vector(7 downto 0);

	signal sram_control_region_active : boolean;
begin
	leds <= led_signal;
	userlogic_reset <= userlogic_reset_signal;

	-- assign sram interface to correct ul or mw interface
	sram_control_region_active <= (unsigned(sram_address) <= unsigned(control_region));
	sram_data_out <= 
		middleware_data_out when sram_control_region_active else
		userlogic_data_out;
	userlogic_wr <= sram_wr when not sram_control_region_active else
		'0';
	userlogic_rd <= sram_rd when not sram_control_region_active else
		'0';
    userlogic_address <= std_logic_vector(unsigned(sram_address) - control_region-1) when not sram_control_region_active else
        (others=>'0');
    
	userlogic_data_in <= sram_data_in;

	-- main data receiving process
	process (reset, clk, sram_rd, sram_wr) 
		variable data_var : std_logic_vector(7 downto 0);
		variable wr_was_low : boolean := false;
	begin
        if rising_edge(clk) then
			if reset = '1' then
                icap_address.ready <= '0';
                icap_address.data <= (others => '0');
                led_signal <= (others => '0');
                userlogic_reset_signal <= '1';
                middleware_data_out <= (others => '0');
            else
				if sram_rd = '1' or sram_wr = '1' then -- or wr_was_low then
					-- writing to an address
					-- only respond when sram_wr goes high again
					if sram_wr = '1' then

						-- control region
						if unsigned(sram_address) <= control_region then
							-- icap
							case unsigned(sram_address) is
							when MULTIBOOT =>
								icap_address.data(7 downto 0) <= unsigned(sram_data_in(7 downto 0));
							when MULTIBOOT + 1 =>
								icap_address.data(15 downto 8) <= unsigned(sram_data_in(7 downto 0));
							when MULTIBOOT + 2 =>
								icap_address.data(23 downto 16) <= unsigned(sram_data_in(7 downto 0));
								icap_address.ready <= '1'; -- will go low automatically when done with multiboot
							when LED =>
								data_var := std_logic_vector(sram_data_in);
								led_signal <= data_var(3 downto 0);
							when USERLOGIC_CONTROL =>
								data_var := std_logic_vector(sram_data_in);
								userlogic_reset_signal <= data_var(0);
							when others =>
							end case;
						end if;
					-- otherwise reading
					else
						-- control region
						if unsigned(sram_address) <= control_region then
							-- write unaffected as zero
							middleware_data_out <= (others => '0');
							
							-- icap
							case unsigned(sram_address) is
							-- -- Only for debug purpose
							-- when MULTIBOOT =>
							-- 	sram_data_out <= icap_address.data(7 downto 0);
							-- when MULTIBOOT + 1 =>
							-- 	sram_data_out <= icap_address.data(15 downto 8);
							-- when MULTIBOOT + 2 =>
							-- 	sram_data_out <= icap_address.data(23 downto 16);
							when LED =>
								middleware_data_out(3 downto 0) <= led_signal;
							when USERLOGIC_CONTROL =>
								middleware_data_out(0) <= userlogic_reset_signal;
							when others =>
								middleware_data_out(7 downto 0) <= sram_address(7 downto 0);
							end case;
							
						end if;
					end if;

				end if;
			
			end if;
		end if;
	end process;
end rtl;
